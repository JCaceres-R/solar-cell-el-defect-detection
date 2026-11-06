"""Fase 2-3 · Corre el preprocesamiento (src/preproc.py) sobre todas las celdas.

Uso:   python scripts/01_preprocesar_todo.py            (todas las celdas, ~3-5 min con 2 núcleos)
       LIMITE=40 python scripts/01_preprocesar_todo.py  (prueba rápida con las primeras 40 celdas)

Salidas en data/interim/ (NO se versionan; se regeneran con este script):
  features.csv   características escalares por celda + idx, tipo, etiqueta, panel, grupo, split
  X_pre.npy      entrada de la CNN: (N, 3, 128, 128) uint8 = [celda sin busbars, déficit, crestas]
  X_raw.npy      celda original reducida a 128x128 (para comparar "con" vs "sin" preprocesamiento)
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

from datos import cargar_imagenes
from preproc import cell_masks, features, image_mask, preprocess

OUT = os.path.join(RAIZ, "data", "interim"); os.makedirs(OUT, exist_ok=True)
N_JOBS = int(os.environ.get("N_JOBS", -1))
LIMITE = int(os.environ.get("LIMITE", 0))

part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
# las plantillas de máscara por tipo se calculan SOLO con entrenamiento (no mirar val/test)
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])
idx = np.arange(len(ims)) if not LIMITE else np.arange(LIMITE)


def run(i):
    m = image_mask(ims[i], T[tipos[i]])
    pp = preprocess(ims[i], m)
    f, _ = features(pp, m)
    a = np.clip(pp["nobus"], 0, 1.2) / 1.2
    b = np.clip(pp["dark"] * 2, 0, 1)
    c = np.clip(pp["ridge"] / 0.15, 0, 1)
    x = np.stack([cv2.resize(z.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) for z in (a, b, c)])
    f["mask_frac"] = float(m.mean())
    return f, (x * 255).astype(np.uint8)


t0 = time.time()
R = Parallel(n_jobs=N_JOBS)(delayed(run)(i) for i in idx)
F = pd.DataFrame([r[0] for r in R])
F = pd.concat([part.iloc[idx].reset_index(drop=True), F], axis=1)
sufijo = f"_prueba{LIMITE}" if LIMITE else ""
F.to_csv(os.path.join(OUT, f"features{sufijo}.csv"), index=False)
np.save(os.path.join(OUT, f"X_pre{sufijo}.npy"), np.stack([r[1] for r in R]))
np.save(os.path.join(OUT, f"X_raw{sufijo}.npy"),
        np.stack([cv2.resize(ims[i], (128, 128), interpolation=cv2.INTER_AREA) for i in idx]))
print(f"{len(idx)} celdas en {time.time() - t0:.0f} s -> {OUT}")
