"""Fase 3 · ¿Cuánto separa el preprocesamiento las celdas defectuosas? (línea base sin red neuronal)

Uso:   python scripts/02_evaluar_caracteristicas.py      (requiere 01_preprocesar_todo.py)

- AUC de cada característica, calculado SOLO con entrenamiento, por tipo de celda.
- Regresión logística entrenada en train y medida en VALIDACIÓN.
- PRUEBA NO SE USA AQUÍ: se reserva para la evaluación final del proyecto (ver docs/03_particion.md).

Salidas: reports/metricas/caracteristicas.json, reports/figures/f1_auc_caracteristicas.png
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
va.assign(prob_logistica=pv)[["idx", "tipo", "prob", "defectuosa", "prob_logistica"]].to_csv(
    os.path.join(RAIZ, "data", "interim", "pred_logistica_val.csv"), index=False)
json.dump(res, open(os.path.join(MET, "caracteristicas.json"), "w"), indent=1, default=float)
print(A.round(3)); print(json.dumps(res["logistica_val"], indent=1, default=float))
