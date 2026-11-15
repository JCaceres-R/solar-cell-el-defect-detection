"""Semana 1 · ¿La máscara y la normalización hacen lo que prometen?

Uso:   python scripts/05_medir_mascara_normalizacion.py      (~1 min)

Mide, SOLO con entrenamiento y validación (prueba no se toca):
  1. Normalización: cuánto varía el brillo de las celdas SANAS (etiqueta 0 %) entre paneles, antes y después
     de dividir por el percentil 90 de la celda (ec. 1 de la propuesta). Si funciona, la variación entre paneles
     debe caer mucho: una celda sana queda cerca de 1 sin importar la exposición del panel.
  2. Atajo de exposición: AUC del brillo crudo (entre todas las celdas y dentro de cada panel) frente al normalizado,
     y correlación entre el % de defectos del panel y el brillo de sus celdas sanas.
  3. Máscara: v0 (solo brillo) frente a v1 (con respaldo de la plantilla), y en qué celdas cambia.

Salidas: reports/figures/n1_normalizacion_paneles.png, n2_mascara_ejemplos.png, n3_brillo_atajo_panel.png
         reports/metricas/semana1_mascara_normalizacion.json
"""
import json
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
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from datos import cargar_imagenes
from estilo import COL_TIPO, GRIS, NOMBRE_TIPO, TINTA, TINTA2, aplicar_estilo, guardar
from preproc import cell_masks, image_mask

aplicar_estilo()
FIG = os.path.join(RAIZ, "reports", "figures"); MET = os.path.join(RAIZ, "reports", "metricas")
os.makedirs(MET, exist_ok=True)

part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])                       # plantillas por tipo: solo con entrenamiento
usar = np.where(part.split.isin(["train", "val"]).values)[0]   # prueba no se toca


def medir(i):
    m0 = image_mask(ims[i], T[tipos[i]], respaldo=False)      # v0: solo por brillo (línea base)
    m = image_mask(ims[i], T[tipos[i]])                         # v1: con respaldo de la plantilla
    f = ims[i].astype(np.float32)
    ref = np.percentile(f[m], 90) + 1e-6
    return {"idx": i, "mask_frac_v0": float(m0.mean()), "mask_frac": float(m.mean()),
            "respaldo": bool((m != m0).any()), "igual_plantilla_v0": bool((m0 == T[tipos[i]]).all()),
            "crudo_media": float(f[m].mean()),
            "crudo_p90": float(ref), "norm_mediana": float(np.median(f[m] / ref)), "norm_media": float((f[m] / ref).mean())}


D = pd.DataFrame(Parallel(n_jobs=-1)(delayed(medir)(i) for i in usar))
D = part.iloc[D.idx].reset_index(drop=True).join(D.drop(columns="idx"))

# ---------------------------------------------------------------- 1. variación entre paneles (celdas sanas)
S = D[D.clase4 == 0]
pan = S.groupby("panel_cod").agg(tipo=("tipo", "first"), n=("tipo", "size"),
                                 crudo=("crudo_media", "median"), norm=("norm_mediana", "median"),
                                 crudo_q1=("crudo_media", lambda x: x.quantile(.25)), crudo_q3=("crudo_media", lambda x: x.quantile(.75)),
                                 norm_q1=("norm_mediana", lambda x: x.quantile(.25)), norm_q3=("norm_mediana", lambda x: x.quantile(.75)))
pan = pan[pan.n >= 5].sort_values("crudo")


def cv(x): return float(np.std(x) / np.mean(x))


def rango(x): return float(np.max(x) / np.min(x))


res = {"n_celdas_usadas": int(len(D)), "n_paneles_con_5_sanas": int(len(pan)),
       "entre_paneles_sanas": {
           "crudo": {"cv": cv(pan.crudo), "max_sobre_min": rango(pan.crudo), "min": float(pan.crudo.min()), "max": float(pan.crudo.max())},
           "normalizado": {"cv": cv(pan.norm), "max_sobre_min": rango(pan.norm), "min": float(pan.norm.min()), "max": float(pan.norm.max())},
           "por_tipo": {t: {"cv_crudo": cv(pan[pan.tipo == t].crudo), "cv_norm": cv(pan[pan.tipo == t].norm)} for t in ["mono", "poly"]}}}

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
x = np.arange(len(pan)); col = [COL_TIPO[t] for t in pan.tipo]
for ax, k, tit, yl in [(axs[0], "crudo", "Antes: brillo medio de las celdas sanas (0-255)", "brillo medio"),
                       (axs[1], "norm", "Después: mediana de I / P90 en las mismas celdas", "brillo normalizado")]:
    ax.vlines(x, pan[k + "_q1"], pan[k + "_q3"], color=col, lw=2.2, alpha=.45)
    ax.scatter(x, pan[k], c=col, s=22, zorder=3)
    ax.set_xticks(x, pan.index, rotation=90, fontsize=6.5); ax.set_ylabel(yl)
    r = res["entre_paneles_sanas"]["crudo" if k == "crudo" else "normalizado"]
    ax.set_title(f"{tit}\nvariación entre paneles: CV = {r['cv']:.0%} · máx/mín = {r['max_sobre_min']:.1f}×", loc="left")
axs[0].set_ylim(0, 255); axs[1].set_ylim(0, 1.15); axs[1].axhline(1, color=TINTA2, lw=.7, ls="--")
for t in ["mono", "poly"]:
    axs[0].scatter([], [], c=COL_TIPO[t], s=22, label=NOMBRE_TIPO[t])
axs[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.text(0.01, -0.07, "Paneles de entrenamiento y validación con al menos 5 celdas sanas (etiqueta 0 %), ordenados por su brillo crudo. "
         "Punto = mediana del panel; barra = cuartiles.", fontsize=7.5, color=TINTA2)
guardar(fig, os.path.join(FIG, "n1_normalizacion_paneles.png"))

# ---------------------------------------------------------------- 2. separación por clase (solo entrenamiento)
Tr = D[D.split == "train"]
auc = {}
for t in ["mono", "poly", "todas"]:
    s = Tr if t == "todas" else Tr[Tr.tipo == t]
    auc[t] = {"crudo_media": float(1 - roc_auc_score(s.defectuosa, s.crudo_media)),
              "norm_mediana": float(1 - roc_auc_score(s.defectuosa, s.norm_mediana))}
# AUC del brillo crudo DENTRO de cada panel: quita el efecto de la exposición del panel
a, w = [], []
for _, s in Tr.groupby("panel_cod"):
    if s.defectuosa.nunique() == 2:
        a.append(1 - roc_auc_score(s.defectuosa, s.crudo_media)); w.append(len(s))
auc["crudo_dentro_de_panel"] = float(np.average(a, weights=w)); auc["n_paneles_dentro"] = len(a)
res["auc_train_oscuro_es_defecto"] = auc

# ¿los paneles con más defectos se fotografiaron más oscuros? (exposición del panel = atajo)
from scipy.stats import spearmanr
P = D.groupby("panel_cod").agg(tipo=("tipo", "first"), pdef=("defectuosa", "mean"))
P["brillo_sanas"] = S.groupby("panel_cod").crudo_media.median()
P["n_sanas"] = S.groupby("panel_cod").size()
P = P[P.n_sanas >= 5]
rho = spearmanr(P.pdef, P.brillo_sanas)
res["exposicion_panel"] = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue), "n_paneles": int(len(P))}

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1.3, 1]})
for t in ["mono", "poly"]:
    q = P[P.tipo == t]
    axs[0].scatter(q.brillo_sanas, q.pdef * 100, c=COL_TIPO[t], s=26, label=NOMBRE_TIPO[t], zorder=3)
axs[0].set_xlabel("brillo crudo de las celdas SANAS del panel (mediana)"); axs[0].set_ylabel("% de celdas defectuosas del panel")
axs[0].set_title(f"Los paneles con más defectos se fotografiaron más oscuros\nSpearman ρ = {rho.statistic:.2f} (p = {rho.pvalue:.3f}) · {len(P)} paneles", loc="left")
axs[0].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
nom = ["brillo crudo,\ntodas las celdas", "brillo crudo,\ndentro de cada panel", "brillo normalizado,\ntodas las celdas"]
val = [auc["todas"]["crudo_media"], auc["crudo_dentro_de_panel"], auc["todas"]["norm_mediana"]]
axs[1].barh(nom[::-1], val[::-1], color=["#80393d", TINTA2, "#c9a3a5"][::-1], height=.55)
for k, v in enumerate(val[::-1]): axs[1].text(v + .005, k, f"{v:.2f}", va="center", fontsize=9)
axs[1].axvline(.5, color=TINTA2, lw=.8, ls="--"); axs[1].set_xlim(.45, .85)
axs[1].set_xlabel("AUC en entrenamiento («más oscura = defectuosa»)")
axs[1].set_title("Parte de lo que separa el brillo crudo\nes la exposición del panel, no el defecto", loc="left")
fig.text(0.01, -0.06, "Solo entrenamiento y validación. Paneles con al menos 5 celdas sanas. AUC 0,5 = azar.", fontsize=7.5, color=TINTA2)
fig.subplots_adjust(wspace=0.6)
guardar(fig, os.path.join(FIG, "n3_brillo_atajo_panel.png"))

# ---------------------------------------------------------------- 3. máscara
res["mascara"] = {t: {"plantilla_frac": float(T[t].mean()),
                      "v0_igual_a_plantilla": float(D[D.tipo == t].igual_plantilla_v0.mean()),
                      "v0_mediana_frac": float(D[D.tipo == t].mask_frac_v0.median()), "v0_min": float(D[D.tipo == t].mask_frac_v0.min()),
                      "v1_min": float(D[D.tipo == t].mask_frac.min())} for t in ["mono", "poly"]}
raras = D[D.respaldo].assign(r=lambda d: d.mask_frac_v0 / d.mask_frac).sort_values("r")
res["mascara"]["v0_igual_a_plantilla_todas"] = float(D.igual_plantilla_v0.mean())
res["mascara"]["celdas_con_respaldo"] = {"n": int(len(raras)), "defectuosas": int(raras.defectuosa.sum()),
                                         "por_tipo": raras.tipo.value_counts().to_dict(),
                                         "por_split": raras.split.value_counts().to_dict(), "idx": raras.idx.tolist()}


def ejemplo(tipo):
    """La celda sana de validación con el brillo crudo mediano de su tipo."""
    s = D[(D.tipo == tipo) & (D.clase4 == 0) & (D.split == "val")].sort_values("crudo_media")
    return int(s.iloc[len(s) // 2].idx)


ej = [("mono sana", ejemplo("mono")), ("poly sana", ejemplo("poly")),
      ("mono, zona apagada", int(D[(D.tipo == "mono") & (D.clase4 == 3)].sort_values("norm_mediana").iloc[5].idx)),
      ("casi apagada: v0 frente a v1", int(raras.iloc[0].idx))]

fig, axs = plt.subplots(2, 4, figsize=(12, 6.7))
for j, (nom, i) in enumerate(ej):
    m = image_mask(ims[i], T[tipos[i]]); m0 = image_mask(ims[i], T[tipos[i]], respaldo=False)
    f = ims[i].astype(np.float32); ref = np.percentile(f[m], 90)
    rgb = np.dstack([ims[i]] * 3).copy()
    distinta = bool((m0 != m).any())
    if distinta:
        c0, _ = cv2.findContours(m0.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, c0, -1, (232, 60, 60), 3)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnt, -1, (27, 175, 122), 2)
    extra = f"\nv0 (rojo) {m0.mean():.0%} → v1 (verde) {m.mean():.0%}" if distinta else f"\nmáscara {m.mean():.0%}"
    axs[0, j].imshow(rgb)
    axs[0, j].set_title(f"{nom}\n#{i} · {part.clase2[i].lower()}{extra}", loc="left", fontsize=8.5)
    axs[1, j].imshow(np.where(m, f / ref, np.nan), cmap="gray", vmin=0, vmax=1.2)
    axs[1, j].set_title(f"normalizada · P90 crudo = {ref:.0f}", loc="left", fontsize=8.5)
    for a in axs[:, j]:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
fig.subplots_adjust(wspace=0.18, hspace=0.32)
fig.text(0.125, 0.035, "Arriba: máscara final (verde) sobre la celda original; en rojo, la máscara solo por brillo (v0) cuando se descarta.\n"
         "Abajo: la celda dividida por su percentil 90 dentro de la máscara, con escala fija de 0 a 1,2 (fuera de la máscara, en blanco).",
         fontsize=7.5, color=TINTA2)
guardar(fig, os.path.join(FIG, "n2_mascara_ejemplos.png"))

json.dump(res, open(os.path.join(MET, "semana1_mascara_normalizacion.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
