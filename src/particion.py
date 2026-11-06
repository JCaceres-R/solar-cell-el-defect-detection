"""Partición entrenamiento / validación / prueba POR PANEL, y verificación de fuga.

Reglas (ver docs/03_particion.md):
  1. La unidad indivisible es el grupo: el panel, salvo que los 5 paneles poly sin frontera exacta
     forman un solo grupo y los paneles gemelos (mismo panel fotografiado dos veces) van juntos.
  2. El grupo que contiene paneles ambiguos va siempre a entrenamiento (en prueba sería casi toda ella).
  3. Validación y prueba reciben cada una ~15 % de las celdas de cada tipo (mono y poly).
  4. Entre muchas asignaciones al azar (semilla fija) se elige la que deja la distribución de las
     4 etiquetas de cada parte, por tipo, más parecida a la del dataset completo.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

SEMILLA = 2026
N_INTENTOS = 300000     # ~1 min; con 20.000 las partes quedaban menos parecidas (puntaje 0,41 vs 0,30)
FRACCION = 0.15        # fracción objetivo de celdas (de cada tipo) para validación y para prueba
PARTES = ["train", "val", "test"]


def _dist4(df: pd.DataFrame) -> np.ndarray:
    c = np.bincount(df.clase4, minlength=4).astype(float)
    return c / max(c.sum(), 1)


def crear_particion(meta: pd.DataFrame, semilla: int = SEMILLA, n_intentos: int = N_INTENTOS):
    """Devuelve (serie split por celda, dict con el puntaje y la semilla usada).

    Cada intento baraja los grupos libres de cada tipo y llena validación y luego prueba, grupo por
    grupo, hasta cubrir FRACCION de las celdas de ese tipo. Se queda con el intento de menor puntaje.
    """
    grupos = meta.groupby("grupo").agg(tipo=("tipo", "first"), exacto=("panel_exacto", "all"),
                                       n=("idx", "size")).reset_index()
    cnt = np.stack([np.bincount(meta.clase4[meta.grupo == g], minlength=4) for g in grupos.grupo]).astype(float)
    libres = {t: np.where((grupos.tipo == t) & grupos.exacto)[0] for t in ["mono", "poly"]}
    meta_t = {t: FRACCION * (meta.tipo == t).sum() for t in ["mono", "poly"]}
    ref = {t: _dist4(meta[meta.tipo == t]) for t in ["mono", "poly"]}
    n_total = len(meta); ng = grupos.n.values

    rng = np.random.default_rng(semilla)
    mejor = None
    for _ in range(n_intentos):
        elegidos = {"val": {}, "test": {}}
        for t in ["mono", "poly"]:
            perm = rng.permutation(libres[t]); k = 0
            for p in ["val", "test"]:
                sel, acum = [], 0
                while acum < meta_t[t] and k < len(perm):
                    sel.append(perm[k]); acum += ng[perm[k]]; k += 1
                elegidos[p][t] = np.array(sel, dtype=int)
        puntaje = 0.0
        for p in ["val", "test"]:
            n_p, n67 = 0.0, 0.0
            for t in ["mono", "poly"]:
                c = cnt[elegidos[p][t]].sum(0)
                puntaje += np.abs(c / c.sum() - ref[t]).sum()
                n_p += c.sum(); n67 += c[2]
            puntaje += 2 * abs(n_p / n_total - FRACCION)
            if n67 < 10:                          # que haya ejemplos de la clase rara (67 %)
                puntaje += 10
        if mejor is None or puntaje < mejor[0]:
            mejor = (puntaje, elegidos)
    asign = {g: "train" for g in grupos.grupo}
    for p in ["val", "test"]:
        for t in ["mono", "poly"]:
            for i in mejor[1][p][t]:
                asign[grupos.grupo[i]] = p
    split = meta.grupo.map(asign).rename("split")
    return split, {"semilla": semilla, "intentos": n_intentos, "puntaje": round(float(mejor[0]), 4)}


def resumen(meta: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Tabla por parte: paneles, celdas, % del total, por tipo y por clase."""
    m = meta.assign(split=split.values)
    filas = []
    for p in PARTES + ["total"]:
        s = m if p == "total" else m[m.split == p]
        fila = {"parte": p, "paneles": s.panel.nunique(), "celdas": len(s),
                "% celdas": round(100 * len(s) / len(m), 1),
                "mono": int((s.tipo == "mono").sum()), "poly": int((s.tipo == "poly").sum())}
        for k, n in enumerate(["0 %", "33 %", "67 %", "100 %"]):
            fila[n] = int((s.clase4 == k).sum())
        fila["FUNCIONAL"] = int((s.defectuosa == 0).sum())
        fila["DEFECTUOSA"] = int((s.defectuosa == 1).sum())
        fila["% defectuosa"] = round(100 * s.defectuosa.mean(), 1)
        for t in ["mono", "poly"]:
            fila[f"% def. {t}"] = round(100 * s[s.tipo == t].defectuosa.mean(), 1)
        filas.append(fila)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ celdas casi idénticas (fuga)
def matriz_similitud(ims: np.ndarray, tipos: np.ndarray, lado: int = 64) -> np.ndarray:
    """Correlación de Pearson entre celdas, tras restar la celda media de su tipo.

    Sin restar la media, todas las celdas se parecen por los busbars; el residuo deja la textura
    propia de cada oblea (granos del silicio, defectos), que es lo que delata a las 'gemelas'.
    """
    sm = np.stack([cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA).astype(np.float32) for im in ims])
    for t in np.unique(tipos):
        sm[tipos == t] -= sm[tipos == t].mean(0)
    v = sm.reshape(len(sm), -1)
    v = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)
    C = v @ v.T / v.shape[1]
    np.fill_diagonal(C, -1)
    return C


def pares_casi_identicos(C: np.ndarray, excluir: np.ndarray, umbral: float = 0.95) -> np.ndarray:
    """Pares (i, j), i < j, con correlación > umbral; se excluyen celdas apagadas (todas se parecen)."""
    pr = np.argwhere(np.triu(C > umbral))
    return pr[~(excluir[pr[:, 0]] | excluir[pr[:, 1]])]


def fuga(pares: np.ndarray, split: np.ndarray) -> dict:
    """Cuántos pares casi idénticos quedan repartidos entre partes distintas."""
    a, b = split[pares[:, 0]], split[pares[:, 1]]
    cruzados = a != b
    out = {"pares": int(len(pares)), "pares_entre_partes": int(cruzados.sum())}
    for p in ["val", "test"]:
        out[f"pares_train_{p}"] = int((((a == "train") & (b == p)) | ((a == p) & (b == "train"))).sum())
    return out


def similitud_entre_paneles(C: np.ndarray, P: pd.DataFrame) -> pd.DataFrame:
    """Para cada par de paneles del mismo tipo: mediana de la correlación celda a celda en la MISMA
    posición (diag) y entre posiciones distintas (fuera). diag alta y fuera baja = mismo panel físico."""
    filas = []
    for i, a in P.iterrows():
        for _, b in P.loc[i + 1:].iterrows():
            if a.tipo != b.tipo:
                continue
            n = min(a.n_celdas, b.n_celdas)
            sub = C[np.ix_(np.arange(a.inicio, a.inicio + n), np.arange(b.inicio, b.inicio + n))]
            fuera = sub[~np.eye(n, dtype=bool)]
            filas.append({"panel_a": a.panel_cod, "panel_b": b.panel_cod, "tipo": a.tipo,
                          "r_misma_pos": round(float(np.median(np.diag(sub))), 3),
                          "r_otra_pos": round(float(np.median(fuera)), 3),
                          "misma_pos_es_la_mejor": int((sub.argmax(1) == np.arange(n)).sum()), "n": n})
    return pd.DataFrame(filas).sort_values("r_misma_pos", ascending=False).reset_index(drop=True)
