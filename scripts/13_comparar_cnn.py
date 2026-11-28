"""Semana 3 · Junta las 30 corridas de la CNN v2 (6 variantes × 5 semillas), aplica el criterio 3 y decide entre A y B.
Figuras c1-c3.

Uso:   python scripts/13_comparar_cnn.py        (requiere scripts/12_cnn.py todas)

Regla de decisión, fijada ANTES de ver los resultados (tabla 3 de la propuesta):
  · Criterio 3 (decide): AUC en validación de la CNN, promedio de las 5 semillas, entre A-global, A-filas y B.
  · Gana la de mayor AUC solo si el intervalo bootstrap del 95 % de la diferencia con la segunda no incluye el 0
    (predicción media de las 5 semillas; 2.000 remuestreos de celdas de validación, por clase).
  · Si lo incluye, es empate en el criterio 3 y deciden los criterios de apoyo 1, 2 y 4 (semana 2).
  · Dedos: B-dedos se adopta solo si supera a B con la misma regla. Si no, no se quitan.

Salidas: reports/figures/c1_entrenamiento.png, c2_roc_val.png, c3_criterios.png y reports/metricas/semana3_cnn.json
"""
import glob
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
from sklearn.metrics import roc_auc_score, roc_curve

from estilo import GRIS, TINTA, TINTA2, aplicar_estilo, guardar

aplicar_estilo()
FIG, MET = os.path.join(RAIZ, "reports", "figures"), os.path.join(RAIZ, "reports", "metricas")
CNN = os.path.join(MET, "cnn")
VARIANTES = ["original", "In", "A-global", "A-filas", "B", "B-dedos"]
NOMBRE = {"original": "Celda cruda", "In": "Mapas sin quitar busbars", "A-global": "A-global (notch)",
          "A-filas": "A-filas (notch en filas)", "B": "B (cierre)", "B-dedos": "B + dedos 1×3"}
COL = {"original": "#8b8a85", "In": "#3d3c39", "A-global": "#7a5bd0", "A-filas": "#b97a00", "B": "#00846f", "B-dedos": "#e05a7a"}
TIPOS = ("todas", "mono", "poly")

N_SEM = 5   # semillas 0-4, las de scripts/12_cnn.py
R = {v: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CNN, f"{v}_s[0-9].json")))] for v in VARIANTES}
P = {v: [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(CNN, f"{v}_s[0-9]_pred.csv")))] for v in VARIANTES}
assert all(len(R[v]) == N_SEM and len(P[v]) == N_SEM for v in VARIANTES), {v: (len(R[v]), len(P[v])) for v in VARIANTES}
assert all(r.get("red") == "v2" for v in VARIANTES for r in R[v]), "hay corridas que no son de la red v2: no se mezclan"

res = {"variantes": {}, "ensamble": {}}
for v in VARIANTES:
    res["variantes"][v] = {t: {m: {"media": float(np.mean([r["val"][t][m] for r in R[v]])), "sd": float(np.std([r["val"][t][m] for r in R[v]], ddof=1)),
                                   "semillas": [r["val"][t][m] for r in R[v]]}
                               for m in ("auc", "exactitud_balanceada", "f1_defectuosa")} for t in TIPOS}
    res["variantes"][v]["mejor_epoca"] = [r["mejor_epoca"] for r in R[v]]
    res["variantes"][v]["ms_por_celda_cnn"] = float(np.mean([r["ms_por_celda_cnn"] for r in R[v]]))

base = P["B"][0][["idx", "tipo", "prob"]].copy(); y = (base.prob >= .5).values
pm = {v: np.mean([d.set_index("idx").loc[base.idx, "pred"].values for d in P[v]], axis=0) for v in VARIANTES}
for v in VARIANTES:
    res["ensamble"][v] = {t: float(roc_auc_score(y[s], pm[v][s])) for t, s in
                          (("todas", np.ones(len(y), bool)), ("mono", (base.tipo == "mono").values), ("poly", (base.tipo == "poly").values))}

rng = np.random.default_rng(0)
pos, neg = np.where(y)[0], np.where(~y)[0]
BOOT = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(2000)]


def diferencia(a, b):
    """AUC(a) - AUC(b) con la predicción media de las 5 semillas, e intervalo bootstrap del 95 %."""
    d = [roc_auc_score(y[k], pm[a][k]) - roc_auc_score(y[k], pm[b][k]) for k in BOOT]
    return {"dif": float(res["ensamble"][a]["todas"] - res["ensamble"][b]["todas"]),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


comp = {f"{a} - {b}": diferencia(a, b) for a, b in [("A-global", "B"), ("A-filas", "B"), ("A-global", "A-filas"), ("B-dedos", "B"),
                                                     ("B", "In"), ("In", "original"), ("B", "original")]}
res["diferencias_auc"] = comp

# ---------------------------------------------------------------- decisión (criterio 3 y, si hay empate, 1, 2 y 4)
c = sorted(["A-global", "A-filas", "B"], key=lambda v: -res["variantes"][v]["todas"]["auc"]["media"])
d = diferencia(c[0], c[1])
if d["ic95"][0] > 0:
    eleccion, motivo = c[0], f"criterio 3: {c[0]} supera a {c[1]} con IC95 {d['ic95']}"
else:
    s2 = json.load(open(os.path.join(MET, "semana2_busbars.json"), encoding="utf-8"))
    cv, tm = s2["criterios_val"]["todas"], s2["tiempo_ms"]
    votos = {v: (cv[v]["c1"] == min(cv[w]["c1"] for w in c)) + (cv[v]["c2"] == min(cv[w]["c2"] for w in c)) + (tm[v] == min(tm[w] for w in c)) for v in c}
    eleccion = max(c, key=lambda v: (votos[v], res["variantes"][v]["todas"]["auc"]["media"]))
    motivo = f"empate en el criterio 3 ({c[0]} - {c[1]}: IC95 {d['ic95']}); criterios 1, 2 y 4: {votos}"
dd = comp["B-dedos - B"]
res["decision"] = {"metodo_busbars": eleccion, "motivo": motivo, "orden_criterio3": c,
                   "quitar_dedos": bool(dd["ic95"][0] > 0), "motivo_dedos": f"B-dedos - B: {dd['dif']:+.4f}, IC95 {dd['ic95']}"}
json.dump(res, open(os.path.join(MET, "semana3_cnn.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"auc_media": {v: {t: round(res["variantes"][v][t]["auc"]["media"], 4) for t in TIPOS} for v in VARIANTES},
                  "auc_sd": {v: round(res["variantes"][v]["todas"]["auc"]["sd"], 4) for v in VARIANTES},
                  "ensamble": res["ensamble"], "diferencias": comp, "decision": res["decision"]}, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- c1: curvas de entrenamiento (media de las 5 semillas)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
for v in VARIANTES:
    H = np.array([[[h["train"], h["val"], h["val_auc"]] for h in r["historial"]] for r in R[v]])
    ep = np.arange(H.shape[1]) + 1
    axs[0].plot(ep, H[:, :, 1].mean(0), color=COL[v], lw=1.6, label=NOMBRE[v])
    axs[0].plot(ep, H[:, :, 0].mean(0), color=COL[v], lw=.9, ls=":")
    m, s = H[:, :, 2].mean(0), H[:, :, 2].std(0)
    axs[1].plot(ep, m, color=COL[v], lw=1.6); axs[1].fill_between(ep, m - s, m + s, color=COL[v], alpha=.12, lw=0)
axs[0].set_ylim(0, None); axs[0].set_xlabel("época"); axs[0].set_ylabel("error cuadrático ponderado")
axs[0].set_title("Pérdida: validación (línea) y entrenamiento (puntos)", loc="left"); axs[0].legend(frameon=False, fontsize=7.5, ncol=2)
axs[1].set_xlabel("época"); axs[1].set_ylabel("AUC en validación"); axs[1].set_ylim(.6, 1)
axs[1].set_title("AUC en validación por época (media ± 1 sd, 5 semillas)", loc="left")
guardar(fig, os.path.join(FIG, "c1_entrenamiento.png"))

# ---------------------------------------------------------------- c2: ROC en validación (predicción media de las 5 semillas)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.1))
for a, (t, s) in zip(axs, (("todas", np.ones(len(y), bool)), ("mono", (base.tipo == "mono").values), ("poly", (base.tipo == "poly").values))):
    for v in VARIANTES:
        fp, tp, _ = roc_curve(y[s], pm[v][s])
        a.plot(fp, tp, color=COL[v], lw=1.8 if v in ("A-global", "A-filas", "B") else 1.1,
               ls="--" if v in ("original", "In") else "-", label=f"{NOMBRE[v]} · {res['ensamble'][v][t]:.3f}")
    a.plot([0, 1], [0, 1], color=GRIS, lw=.8)
    a.set_title({"todas": "Todas", "mono": "Monocristalinas", "poly": "Policristalinas"}[t] + f" (n = {s.sum()})", loc="left")
    a.set_xlabel("falsos positivos"); a.set_aspect("equal"); a.legend(frameon=False, fontsize=6.8, loc="lower right")
axs[0].set_ylabel("verdaderos positivos")
guardar(fig, os.path.join(FIG, "c2_roc_val.png"))

# ---------------------------------------------------------------- c3: los 4 criterios de la tabla 3
s2 = json.load(open(os.path.join(MET, "semana2_busbars.json"), encoding="utf-8"))
cv, tm = s2["criterios_val"]["todas"], s2["tiempo_ms"]
filas = []
for v in ("A-global", "A-filas", "B"):
    a = res["variantes"][v]["todas"]["auc"]
    filas.append([NOMBRE[v], f"{cv[v]['c1']:.1f} %", f"{cv[v]['c2']:.1f} %", f"{a['media']:.3f} ± {a['sd']:.3f}", f"{tm[v]:.1f} ms"])
fig, ax = plt.subplots(figsize=(10, 1.9)); ax.axis("off")
tb = ax.table(cellText=filas, colLabels=["Método", "1 · Residual en busbar ↓", "2 · Cambio fuera ↓", "3 · AUC CNN (decide) ↑", "4 · Tiempo ↓"],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor(GRIS)
    if r_ == 0: cell.set_text_props(weight="bold", color=TINTA); cell.set_facecolor("#f1efe9")
    elif filas[r_ - 1][0] == NOMBRE[eleccion]: cell.set_facecolor("#e3f4ef")
    if c_ == 3 and r_ > 0: cell.set_text_props(weight="bold")
ax.set_title(f"Criterios de la tabla 3 en validación · elegido: {NOMBRE[eleccion]}", loc="left", color=TINTA)
guardar(fig, os.path.join(FIG, "c3_criterios.png"))
