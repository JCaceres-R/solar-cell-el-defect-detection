"""Fase 2-5 · Figura del pipeline paso a paso y ejemplos de errores (en VALIDACIÓN).

Uso:   python scripts/03_figuras_pipeline.py   (los errores requieren 02_evaluar_caracteristicas.py)

Salidas: reports/figures/f2_pipeline_pasos.png, reports/figures/f3_errores_val.png
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import cargar_imagenes
from estilo import aplicar_estilo, guardar
from preproc import cell_masks, features, image_mask, preprocess

aplicar_estilo()
FIG = os.path.join(RAIZ, "reports", "figures")
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])


def overlay(img, pp, cr):
    """Mapa de defectos: zonas apagadas en naranja, grietas en rojo, sobre la celda original."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    col = rgb.copy(); col[pp["dark"] > 0.15] = (255, 170, 0); col[cr] = (255, 40, 40)
    return np.clip(.45 * rgb + .55 * col, 0, 255).astype(np.uint8)


def correr(k):
    m = image_mask(ims[k], T[tipos[k]]); pp = preprocess(ims[k], m); f, cr = features(pp, m)
    return m, pp, cr


# 1. pipeline paso a paso en 3 celdas de ENTRENAMIENTO: grieta mono, zona inactiva mono, defecto poly
trd = part[(part.split == "train") & (part.clase4 == 3)]
sel = [(int(trd[trd.tipo == "mono"].idx.iloc[5]), "Defecto (mono)"),
       (int(trd[trd.tipo == "mono"].idx.iloc[12]), "Defecto (mono)"),
       (int(trd[trd.tipo == "poly"].idx.iloc[40]), "Defecto (poly)")]
fig, axs = plt.subplots(3, 6, figsize=(11.5, 6.3))
for i, (k, nombre) in enumerate(sel):
    m, pp, cr = correr(k)
    rgbm = cv2.cvtColor(ims[k], cv2.COLOR_GRAY2RGB); rgbm[~m] = (rgbm[~m] * .35).astype(np.uint8)
    rb = cv2.cvtColor(ims[k], cv2.COLOR_GRAY2RGB)
    for r in pp["peaks"]: cv2.line(rb, (0, r), (299, r), (0, 160, 140), 2)
    pasos = [(ims[k], "1. Original", "gray"), (rgbm, "2. Máscara", None), (rb, "3. Busbars detectados", None),
             (np.clip(pp["nobus"], .3, 1.2), "4. Normalizada, sin busbars", "gray"),
             (pp["ridge"], "5. Crestas (Sato)", "magma"), (overlay(ims[k], pp, cr), "6. Mapa de defectos", None)]
    for j, (im, tt, cm) in enumerate(pasos):
        axs[i, j].imshow(im, cmap=cm); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        if i == 0: axs[i, j].set_title(tt, loc="left", fontsize=7.6)
    axs[i, 0].set_ylabel(f"{nombre}\n#{k} · P={part.prob[k]:.2f}", fontsize=8)
fig.subplots_adjust(wspace=.05, hspace=.06)
guardar(fig, os.path.join(FIG, "f2_pipeline_pasos.png"))

# 2. errores de la línea base logística en VALIDACIÓN
ruta = os.path.join(RAIZ, "data", "interim", "pred_logistica_val.csv")
if os.path.exists(ruta):
    V = pd.read_csv(ruta)
    fp = V[(V.prob_logistica > .5) & (V.defectuosa == 0)].sort_values("prob_logistica", ascending=False).head(5)
    fn = V[(V.prob_logistica < .5) & (V.defectuosa == 1)].sort_values("prob_logistica").head(5)
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.8))
    for r, (D, lab) in enumerate([(fp, "Falso positivo"), (fn, "Falso negativo")]):
        for j, (_, row) in enumerate(D.iterrows()):
            k = int(row.idx); m, pp, cr = correr(k)
            axs[r, j].imshow(overlay(ims[k], pp, cr))
            axs[r, j].set_title(f"{lab} · {row.tipo}\netiqueta {row.prob:.2f} · modelo {row.prob_logistica:.2f}",
                                loc="left", fontsize=7.6)
        for a in axs[r]: a.set_xticks([]); a.set_yticks([])
    fig.subplots_adjust(wspace=.05, hspace=.28)
    guardar(fig, os.path.join(FIG, "f3_errores_val.png"))
print("ok")
