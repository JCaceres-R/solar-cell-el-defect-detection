"""Semana 3 · Entrada de la CNN para cada variante del preprocesamiento (§4.5). Solo entrenamiento y validación.

Uso:   python scripts/11_tensores_cnn.py          (~2 min)
       TEST=1 python scripts/11_tensores_cnn.py   (SEMANA 4, con el sistema ya congelado: agrega la prueba)

Variantes (cada una, 3 canales de 128 × 128 en uint8):
  original   la celda cruda repetida en los 3 canales: la CNN sin preprocesamiento
  In         [In, D, crestas] sin quitar busbars: mide cuánto aportan los mapas por sí solos
  A-global   [Ic, D, crestas] con el notch en toda la celda
  A-filas    [Ic, D, crestas] con el notch solo en las filas de busbar
  B          [Ic, D, crestas] con el cierre vertical
  B-dedos    como B, más el cierre horizontal de 1×3 de los dedos (para confirmar que no conviene)

Los parámetros de los mapas son las constantes de src/mapas.py, fijadas en entrenamiento por scripts/10_mapas.py.
Salidas en data/interim/ (no se versionan): cnn_indices.csv y X_<variante>.npy
"""
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import busbars as B
import mapas as Mp
from datos import cargar_imagenes
from preproc import cell_masks, image_mask

OUT = os.path.join(RAIZ, "data", "interim"); os.makedirs(OUT, exist_ok=True)
VARIANTES = ["original", "In", "A-global", "A-filas", "B", "B-dedos"]
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
partes = ["train", "val"] + (["test"] if os.environ.get("TEST") == "1" else [])
ims = cargar_imagenes(); tipos = part.tipo.values
T = cell_masks(ims[(part.split == "train").values], tipos[(part.split == "train").values])
idx = np.where(part.split.isin(partes).values)[0]


def run(i):
    m = image_mask(ims[i], T[tipos[i]]); f = ims[i].astype(np.float32)
    In = f / (np.percentile(f[m], 90) + 1e-6); bb = B.detectar(In, m)
    out = {"original": np.repeat(cv2.resize(ims[i], (128, 128), interpolation=cv2.INTER_AREA)[None], 3, 0)}
    for v in VARIANTES[1:]:
        Ic = Mp.limpiar(In, m, bb, "B" if v == "B-dedos" else v, dedos=3 if v == "B-dedos" else 1)
        out[v] = Mp.entrada_cnn(Ic, Mp.deficit(Ic, m), Mp.sato_crestas(Ic, m))
    return out


t0 = time.time()
R = Parallel(n_jobs=-1)(delayed(run)(int(i)) for i in idx)
part.iloc[idx].to_csv(os.path.join(OUT, "cnn_indices.csv"), index=False)
for v in VARIANTES:
    np.save(os.path.join(OUT, f"X_{v}.npy"), np.stack([r[v] for r in R]))
print(f"{len(idx)} celdas ({', '.join(partes)}) × {len(VARIANTES)} variantes en {time.time() - t0:.0f} s -> {OUT}")
