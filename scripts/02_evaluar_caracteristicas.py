"""Fase 3 · ¿Cuánto separa el preprocesamiento las celdas defectuosas? (línea base sin red neuronal)

Uso:   python scripts/02_evaluar_caracteristicas.py      (requiere 01_preprocesar_todo.py)

- AUC de cada característica, calculado SOLO con entrenamiento, por tipo de celda.
- Regresión logística entrenada en train y medida en VALIDACIÓN.
- PRUEBA NO SE USA AQUÍ: se reserva para la evaluación final del proyecto (ver docs/03_particion.md).

Salidas: reports/metricas/caracteristicas.json, reports/figures/f1_auc_caracteristicas.png, reports/figures/f4_linea_base_val.png
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from estilo import COL_TIPO, NOMBRE_TIPO, TINTA2, aplicar_estilo, guardar

aplicar_estilo()
F = pd.read_csv(os.path.join(RAIZ, "data", "interim", "features.csv"))
MET = os.path.join(RAIZ, "reports", "metricas"); os.makedirs(MET, exist_ok=True)
FEAT = ["area_oscura_15", "area_oscura_30", "area_oscura_50", "deficit_medio", "deficit_p99",
        "long_grietas", "grieta_max", "cresta_media", "std_norm", "brillo_ref"]
NOMBRE = {"area_oscura_15": "Área con déficit > 15 %", "area_oscura_30": "Área con déficit > 30 %",
          "area_oscura_50": "Área con déficit > 50 %", "deficit_medio": "Déficit medio", "deficit_p99": "Déficit p99",
          "long_grietas": "Longitud total de grietas", "grieta_max": "Grieta más larga",
          "cresta_media": "Respuesta media de crestas", "std_norm": "Desv. estándar (sin busbars)",
          "brillo_ref": "Brillo de referencia (crudo)"}
tr, va = F[F.split == "train"], F[F.split == "val"]

# 1. AUC por característica (solo train)
A = pd.DataFrame({t: {f: max(a := roc_auc_score(S.defectuosa, S[f]), 1 - a) for f in FEAT}
                  for t, S in [("mono", tr[tr.tipo == "mono"]), ("poly", tr[tr.tipo == "poly"]), ("todas", tr)]})
fig, ax = plt.subplots(figsize=(7.5, 3.8))
orden = A.sort_values("todas").index; yy = np.arange(len(orden))
for k, t in enumerate(["mono", "poly"]):
    ax.barh(yy + (k - .5) * .4, A.loc[orden, t], .37, color=COL_TIPO[t], label=NOMBRE_TIPO[t])
ax.set_yticks(yy, [NOMBRE[o] for o in orden]); ax.axvline(.5, color=TINTA2, lw=.8, ls="--")
ax.set_xlim(.45, 1); ax.set_xlabel("AUC en entrenamiento (0,5 = azar · 1 = separación perfecta)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Cada característica, sola, para separar defectuosa vs. funcional", loc="left")
guardar(fig, os.path.join(RAIZ, "reports", "figures", "f1_auc_caracteristicas.png"))

# 2. regresión logística: train -> val
def X(S):
    x = S[FEAT + ["n_busbars"]].copy(); x["poly"] = (S.tipo == "poly").astype(int); return x
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(X(tr), tr.defectuosa)
pv = clf.predict_proba(X(va))[:, 1]; yv = va.defectuosa.values; yh = (pv > .5).astype(int)
res = {"auc_por_caracteristica_train": A.round(3).to_dict(),
       "logistica_val": {"auc": roc_auc_score(yv, pv), "exactitud": float((yh == yv).mean()),
                         "exactitud_balanceada": balanced_accuracy_score(yv, yh), "f1_defectuosa": f1_score(yv, yh),
                         "exactitud_mono": float((yh == yv)[va.tipo.values == "mono"].mean()),
                         "exactitud_poly": float((yh == yv)[va.tipo.values == "poly"].mean()),
                         "matriz_confusion": confusion_matrix(yv, yh).tolist()}}


# 3. las mismas métricas por tipo de celda (la propuesta pide reportar mono y poly por separado)
def met(y, p):
    h = (p > .5).astype(int)
    return {"n": int(len(y)), "n_defectuosas": int(y.sum()), "auc": roc_auc_score(y, p),
            "exactitud": float((h == y).mean()), "exactitud_balanceada": balanced_accuracy_score(y, h),
            "f1_defectuosa": f1_score(y, h), "matriz_confusion": confusion_matrix(y, h, labels=[0, 1]).tolist()}


tv = va.tipo.values
res["logistica_val_por_tipo"] = {t: met(yv[tv == t], pv[tv == t]) for t in ["mono", "poly"]}
res["logistica_val_por_tipo"]["todas"] = met(yv, pv)

# 4. la misma logística SIN el brillo crudo (brillo_ref): el brillo crudo arrastra la exposición del panel
#    (ver scripts/05_medir_mascara_normalizacion.py), así que se reporta también sin él
F2 = [f for f in FEAT if f != "brillo_ref"]
def X2(S):
    x = S[F2 + ["n_busbars"]].copy(); x["poly"] = (S.tipo == "poly").astype(int); return x
clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X2(tr), tr.defectuosa)
pv2 = clf2.predict_proba(X2(va))[:, 1]
res["logistica_val_sin_brillo_crudo"] = {t: met(yv[tv == t], pv2[tv == t]) for t in ["mono", "poly"]}
res["logistica_val_sin_brillo_crudo"]["todas"] = met(yv, pv2)

from sklearn.metrics import roc_curve
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
for t in ["mono", "poly"]:
    fp, tp_, _ = roc_curve(yv[tv == t], pv[tv == t])
    axs[0].plot(fp, tp_, color=COL_TIPO[t], lw=1.8, label=f"{NOMBRE_TIPO[t]} · AUC {res['logistica_val_por_tipo'][t]['auc']:.2f}")
axs[0].plot([0, 1], [0, 1], color=TINTA2, lw=.8, ls="--")
axs[0].set_xlabel("falsos positivos (funcionales marcadas)"); axs[0].set_ylabel("verdaderos positivos (defectuosas halladas)")
axs[0].legend(frameon=False, loc="lower right"); axs[0].set_title("Curva ROC en validación", loc="left")
axs[0].set_aspect("equal")
for ax, t in zip(axs[1:], ["mono", "poly"]):
    cm = np.array(res["logistica_val_por_tipo"][t]["matriz_confusion"])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues" if t == "mono" else "Oranges", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            fr = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{fr:.0%}", ha="center", va="center", fontsize=9.5, color="white" if fr > .55 else "black")
    ax.set_xticks([0, 1], ["funcional", "defectuosa"]); ax.set_yticks([0, 1], ["funcional", "defectuosa"])
    ax.set_xlabel("predicha"); ax.set_ylabel("real" if t == "mono" else "")
    r = res["logistica_val_por_tipo"][t]
    ax.set_title(f"{NOMBRE_TIPO[t]} · exact. bal. {r['exactitud_balanceada']:.0%}", loc="left")
    for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(wspace=0.55)
guardar(fig, os.path.join(RAIZ, "reports", "figures", "f4_linea_base_val.png"))

va.assign(prob_logistica=pv)[["idx", "tipo", "prob", "defectuosa", "prob_logistica"]].to_csv(
    os.path.join(RAIZ, "data", "interim", "pred_logistica_val.csv"), index=False)
json.dump(res, open(os.path.join(MET, "caracteristicas.json"), "w"), indent=1, default=float)
print(A.round(3)); print(json.dumps(res["logistica_val"], indent=1, default=float))
