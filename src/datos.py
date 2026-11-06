"""Carga del dataset ELPV y tabla de metadatos por celda (incluye el panel reconstruido).

El paquete `elpv-dataset` trae 2.624 imágenes de celdas y, por cada una, solo tres datos:
ruta, probabilidad de defecto y tipo (mono/poly). NO trae el número de panel, que es
necesario para partir el dataset sin fuga. Aquí se reconstruye (ver docs/01_dataset.md, §5).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

N_CELDAS = 2624
CLASES4 = ["0 %", "33 %", "67 %", "100 %"]          # probabilidad de defecto del experto
CLASES2 = ["FUNCIONAL", "DEFECTUOSA"]                 # 0/33 % -> FUNCIONAL ; 67/100 % -> DEFECTUOSA
UMBRAL_OSCURA = 40                                    # brillo medio < 40/255 = celda totalmente apagada

# ---------------------------------------------------------------------------------------------
# Reconstrucción de paneles.
# Las celdas vienen ordenadas por panel (cell0001.png ... cell2624.png). El tipo cambia en 10
# tramos; cada tramo es una sucesión de paneles completos de 60 celdas, salvo los casos anotados.
# Total resultante: 18 paneles mono + 26 poly = 44, igual a lo publicado por los autores.
# Cada entrada: (inicio del tramo, tipo, tamaños de los paneles del tramo en orden).
# ---------------------------------------------------------------------------------------------
TRAMOS = [
    (0,    "mono", [60]),
    (60,   "poly", [60] * 5),
    (360,  "mono", [60] * 3),
    (540,  "poly", [60] * 6),
    # 714 celdas = 11 paneles de 60 + 1 de 54. Los saltos de exposición marcan fronteras en
    # 1020, 1080, 1140, 1200, 1254, 1314, ... -> el panel incompleto es el de las celdas 1200-1253.
    (900,  "mono", [60] * 5 + [54] + [60] * 6),
    # 530 celdas = 8 paneles de 60 + 1 de 50. Fronteras seguras en 1674, 1964, 2024 y 2084; el
    # panel de 50 está entre 1674 y 1963 pero no se puede ubicar con certeza. Esos 5 paneles se
    # marcan como "no exactos" y se tratan como UN solo grupo al partir (ver PANELES_AMBIGUOS).
    (1614, "poly", [60] + [60] * 4 + [50] + [60] * 3),
    (2144, "mono", [72]),   # tramo de 72 celdas: 1 panel (por conteo: 18 paneles mono en total)
    (2216, "poly", [60] * 3),
    (2396, "mono", [48]),   # tramo de 48 celdas: 1 panel
    (2444, "poly", [60] * 3),
]
INICIO_AMBIGUO, FIN_AMBIGUO = 1674, 1964   # celdas [1674, 1964) -> 5 paneles poly sin frontera exacta

# Paneles "gemelos": pares de paneles cuyas celdas, comparadas posición a posición, son casi idénticas
# (mediana de la correlación en la misma posición >= 0,70; entre posiciones distintas es mucho menor).
# Lo más probable es que sean el MISMO panel físico fotografiado dos veces. Deben ir a la misma parte.
# Evidencia: notebooks/01_EDA.ipynb, sección 7, y docs/01_dataset.md.
PANELES_GEMELOS = [("M05", "M08"), ("M11", "M12"), ("M13", "M14"), ("M15", "M16"), ("P14", "P22")]


def ruta_dataset() -> str:
    import elpv_dataset
    return os.path.join(os.path.dirname(elpv_dataset.__file__), "data")


@lru_cache(maxsize=1)
def cargar_imagenes() -> np.ndarray:
    """Devuelve las 2.624 imágenes como un arreglo uint8 de forma (2624, 300, 300)."""
    meta = _leer_labels()
    base = ruta_dataset()
    ims = np.stack([np.asarray(Image.open(os.path.join(base, f))) for f in meta.archivo])
    assert ims.shape == (N_CELDAS, 300, 300) and ims.dtype == np.uint8
    return ims


def _leer_labels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ruta_dataset(), "labels.csv"), sep=r"\s+", header=None,
                     names=["archivo", "prob", "tipo"])
    assert len(df) == N_CELDAS
    return df


def paneles() -> pd.DataFrame:
    """Tabla de los 44 paneles reconstruidos: id, tipo, celda inicial, nº de celdas, exacto."""
    filas, pid = [], 0
    cont = {"mono": 0, "poly": 0}
    for inicio, tipo, tamanos in TRAMOS:
        s = inicio
        for n in tamanos:
            cont[tipo] += 1
            exacto = not (INICIO_AMBIGUO <= s < FIN_AMBIGUO)
            filas.append(dict(panel=pid, panel_cod=f"{tipo[0].upper()}{cont[tipo]:02d}", tipo=tipo,
                              inicio=s, n_celdas=n, exacto=exacto))
            s += n; pid += 1
    P = pd.DataFrame(filas)
    assert P.n_celdas.sum() == N_CELDAS and len(P) == 44
    assert (P.tipo == "mono").sum() == 18 and (P.tipo == "poly").sum() == 26
    return P


def _grupos(P: pd.DataFrame) -> pd.Series:
    """Une paneles que no pueden separarse (bloque ambiguo + gemelos). Devuelve panel_cod -> nombre del grupo."""
    padre = {c: c for c in P.panel_cod}
    def raiz(c):
        while padre[c] != c: c = padre[c]
        return c
    ambiguos = P[~P.exacto].panel_cod.tolist()
    for a, b in list(zip(ambiguos, ambiguos[1:])) + PANELES_GEMELOS:
        padre[raiz(b)] = raiz(a)
    miembros = {}
    for c in P.panel_cod: miembros.setdefault(raiz(c), []).append(c)
    nombre = {r: (m[0] if len(m) == 1 else "+".join(m)) for r, m in miembros.items()}
    return pd.Series({c: nombre[raiz(c)] for c in P.panel_cod})


def metadatos(con_brillo: bool = True) -> pd.DataFrame:
    """Una fila por celda: archivo, etiquetas (4 y 2 clases), tipo, panel, grupo de partición y brillo.

    `grupo` es la unidad indivisible para partir el dataset: el panel, salvo que (a) los 5 paneles poly
    sin frontera exacta forman un único grupo y (b) los paneles gemelos van juntos.
    """
    df = _leer_labels()
    df.insert(0, "idx", np.arange(N_CELDAS))
    df["clase4"] = np.rint(df.prob * 3).astype(int)              # 0, 1, 2, 3
    df["defectuosa"] = (df.prob >= 0.5).astype(int)              # 0 = FUNCIONAL, 1 = DEFECTUOSA
    df["clase2"] = np.array(CLASES2)[df.defectuosa]

    P = paneles()
    pan = np.repeat(P.panel.values, P.n_celdas.values)
    df["panel"] = pan
    df = df.merge(P[["panel", "panel_cod", "exacto"]], on="panel", how="left").rename(
        columns={"exacto": "panel_exacto"})
    df["pos_en_panel"] = df.idx - np.repeat(P.inicio.values, P.n_celdas.values)
    df["grupo"] = _grupos(P).loc[df.panel_cod].values

    # el tipo del panel reconstruido debe coincidir con el de la etiqueta de cada celda
    tipo_panel = np.repeat(P.tipo.values, P.n_celdas.values)
    assert (tipo_panel == df.tipo.values).all(), "la tabla TRAMOS no coincide con los tipos del dataset"

    if con_brillo:
        ims = cargar_imagenes()
        flat = ims.reshape(N_CELDAS, -1)
        df["brillo_medio"] = flat.mean(1).round(2)
        df["brillo_p05"] = np.percentile(flat, 5, axis=1)
        df["brillo_p95"] = np.percentile(flat, 95, axis=1)
        df["frac_saturada"] = (flat >= 255).mean(1).round(5)
        df["oscura"] = (df.brillo_medio < UMBRAL_OSCURA).astype(int)
    return df
