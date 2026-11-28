"""Semana 3 · Mapas de grietas y de zonas apagadas (§4.4). Barrido en ENTRENAMIENTO, reporte en VALIDACIÓN. Figuras m1-m3.

Uso:   python scripts/10_mapas.py        (~4 min; la prueba NO se carga en ningún momento)

ELPV no dice dónde está el defecto, así que los mapas no se miden contra una anotación (§5.2). Para elegir sus
parámetros se usan tres medidas indirectas, todas por tipo de celda:
  · AUC de una sola cifra del mapa (área apagada, largo de grietas) para separar funcional de defectuosa;
  · largo de "grietas" que el mapa marca en celdas sanas (0 %), que debería ser bajo;
  · grietas sintéticas pintadas sobre celdas sanas: fracción que el mapa encuentra.

  1. Déficit D (ec. 5): σ del suavizado y umbral de zona apagada.
  2. Crestas: Sato (ec. 4) a dos juegos de escalas frente al plan B (banco de Gabor), con umbral K·MAD y filtro de
     longitud y rectitud. Aquí se ve si Sato confunde los bordes de grano de las poly con grietas.
  3. Grietas sintéticas en celdas sanas de entrenamiento, mono y poly.

Salidas: reports/figures/m1_mapas.png, m2_poly_granos.png, m3_barrido_mapas.png y reports/metricas/semana3_mapas.json
"""
import json
import os
import sys
import time
from itertools import product

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import busbars as B
import mapas as Mp
from datos import cargar_imagenes
from estilo import COL_TIPO, GRIS, NOMBRE_TIPO, TINTA, TINTA2, aplicar_estilo, guardar
from preproc import cell_masks, image_mask

aplicar_estilo()
FIG, MET = os.path.join(RAIZ, "reports", "figures"), os.path.join(RAIZ, "reports", "metricas")
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
T = cell_masks(ims[(part.split == "train").values], tipos[(part.split == "train").values])
usar = np.where(part.split.isin(["train", "val"]).values)[0]

SIG_D, UMB_D = (2, 4, 8), (.1, .15, .2, .3, .4)
DETECTORES = {"Sato σ 1-2": ("sato", (1, 2)), "Sato σ 1-3": ("sato", (1, 2, 3)),
              "Gabor λ6": ("gabor", (6, 1.5)), "Gabor λ8": ("gabor", (8, 2.0))}
KS, LARGOS, RECT = (4, 6, 8), (18, 30), (0, .8)


def preparar(i):
    m = image_mask(ims[i], T[tipos[i]]); f = ims[i].astype(np.float32)
    In = f / (np.percentile(f[m], 90) + 1e-6)
    return In, m, B.detectar(In, m)


def respuesta(det, Ic, m):
    k, p = DETECTORES[det]
    return Mp.sato_crestas(Ic, m, p) if k == "sato" else Mp.gabor_crestas(Ic, m, lambd=p[0], sigma=p[1])


def medir(i):
    In, m, bb = preparar(i); Ic = Mp.limpiar(In, m, bb, "B"); n = Mp.interior(m)
    d = {"idx": int(i), "split": part.split[i], "tipo": tipos[i], "panel": part.panel_cod[i],
         "y": int(part.prob[i] >= .5), "c4": int(part.clase4[i])}
    for s in SIG_D:
        D = Mp.deficit(Ic, m, s)
        for u in UMB_D:
            d[f"area|{s}|{u}"] = float((D[n] > u).mean())
    for det in DETECTORES:
        R = respuesta(det, Ic, m)
        d[f"p995|{det}"] = float(np.percentile(R[n], 99.5))
        for k, l, r in product(KS, LARGOS, RECT):
            d[f"L|{det}|{k}|{l}|{r}"] = Mp.grietas(R, n, k, l, r)[1]
    return d


t0 = time.time()
CACHE = os.path.join(RAIZ, "data", "interim", "semana3_mapas_por_celda.csv")      # ~12 min; se reusa si existe
if os.path.exists(CACHE):
    X = pd.read_csv(CACHE)
else:
    X = pd.DataFrame(Parallel(n_jobs=-1)(delayed(medir)(int(i)) for i in usar))
    X.to_csv(CACHE, index=False)
print(f"{len(X)} celdas en {time.time() - t0:.0f} s")
TR, VA = X[X.split == "train"], X[X.split == "val"]


def auc(q, c):
    return float(roc_auc_score(q.y, q[c])) if q.y.nunique() == 2 else None


def tabla(q, cols):
    return {c: {t: {"auc": auc(q[q.tipo == t], c), "sanas_media": float(q[(q.tipo == t) & (q.c4 == 0)][c].mean()),
                    "defectuosas_media": float(q[(q.tipo == t) & (q.y == 1)][c].mean())} for t in ("mono", "poly")}
            for c in cols}


res = {"n": {"train": int(len(TR)), "val": int(len(VA))}}

# ---------------------------------------------------------------- 1. déficit, elegido en train
# Regla: un mapa de zonas apagadas debe quedar casi vacío en una celda sana. Por tipo, solo cuentan los umbrales cuya área
# media en las sanas (0 %) de entrenamiento es <= SANA_MAX; entre ellos, el de mayor AUC. σ es común a los dos tipos: el
# que da el mayor AUC medio. Sin esta regla gana un umbral de 0,1 que en las poly mide la textura del grano de cada panel:
# AUC 0,80 en entrenamiento y 0,57 en validación (queda en el json como "sin_regla").
SANA_MAX = .01
cA = [c for c in X.columns if c.startswith("area|")]
tA = tabla(TR, cA)
mejor_tipo = {}
for s in SIG_D:
    ok = {t: [c for c in cA if c.split("|")[1] == str(s) and tA[c][t]["sanas_media"] <= SANA_MAX] for t in ("mono", "poly")}
    if all(ok.values()):
        mejor_tipo[s] = {t: max(ok[t], key=lambda c: tA[c][t]["auc"]) for t in ok}
sD = max(mejor_tipo, key=lambda s: np.mean([tA[mejor_tipo[s][t]][t]["auc"] for t in ("mono", "poly")]))
elegA = mejor_tipo[sD]
sin_regla = max(cA, key=lambda c: (tA[c]["mono"]["auc"] + tA[c]["poly"]["auc"]) / 2)
tV = tabla(VA, cA)
res["deficit"] = {"train": tA, "regla_sana_max": SANA_MAX, "sigma": sD, "umbral": {t: float(elegA[t].split("|")[2]) for t in elegA},
                  "val": {t: tV[elegA[t]][t] for t in elegA}, "sin_regla": {"config": sin_regla, "val": tV[sin_regla]}}
mejorA = elegA

# ---------------------------------------------------------------- 2. crestas: se elige en train por el AUC medio de los dos tipos
cL = [c for c in X.columns if c.startswith("L|")]
tL = tabla(TR, cL)
media = {c: (tL[c]["mono"]["auc"] + tL[c]["poly"]["auc"]) / 2 for c in cL}
mejorL = max(cL, key=media.get)
mejor_det = {d: max([c for c in cL if c.split("|")[1] == d], key=media.get) for d in DETECTORES}
res["crestas"] = {"train": tL, "elegido": mejorL, "mejor_por_detector": mejor_det,
                  "val": tabla(VA, list(mejor_det.values())),
                  "respuesta_p995_sanas_train": {d: {t: float(TR[(TR.tipo == t) & (TR.c4 == 0)][f"p995|{d}"].median())
                                                     for t in ("mono", "poly")} for d in DETECTORES}}
print("déficit elegido", sD, mejorA, "· crestas elegido", mejorL)

# ---------------------------------------------------------------- 3. grietas sintéticas sobre celdas sanas de entrenamiento
yy, xx = np.mgrid[:300, :300].astype(np.float32)


def linea(cx, cy, ang, largo, sigma=1.0):
    a = np.deg2rad(ang); dx, dy = np.cos(a), -np.sin(a)
    t, nn = (xx - cx) * dx + (yy - cy) * dy, -(xx - cx) * dy + (yy - cy) * dx
    return np.exp(-nn ** 2 / (2 * sigma ** 2)) * (np.abs(t) < largo / 2)


CONTRASTES = (.1, .2, .3)


def sintetica(i):
    rng = np.random.default_rng(int(i)); In, m, bb = preparar(i); out = []
    n = Mp.interior(m)
    for c in CONTRASTES:
        ang = rng.uniform(0, 180); cx, cy = rng.uniform(90, 210, 2)
        l = linea(cx, cy, ang, 70); nucleo = (l > .6) & n
        Ic = Mp.limpiar(In * (1 - c * l), m, bb, "B")
        for det, col in mejor_det.items():
            _, _, k, lm, r = col.split("|")
            g = Mp.grietas(respuesta(det, Ic, m), n, float(k), float(lm), float(r))[0]
            out.append({"tipo": tipos[i], "contraste": c, "detector": det, "detectada": bool(g[nucleo].mean() >= .5)})
    return out


sanas = part[(part.split == "train") & (part.clase4 == 0)]
base = list(sanas[sanas.tipo == "mono"].sample(80, random_state=5).index) + list(sanas[sanas.tipo == "poly"].sample(80, random_state=5).index)
S = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(sintetica)(int(i)) for i in base) for x in r])
res["sinteticas"] = {"n_celdas": len(base), "largo_px": 70, "contrastes": list(CONTRASTES),
                     "detectada": {d: {t: S[(S.detector == d) & (S.tipo == t)].groupby("contraste").detectada.mean().round(4).to_dict()
                                       for t in ("mono", "poly")} for d in DETECTORES}}
json.dump(res, open(os.path.join(MET, "semana3_mapas.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("deficit", "crestas")} | {
    "deficit": {"elegido": mejorA, "val": res["deficit"]["val"]},
    "crestas": {"elegido": mejorL, "val": res["crestas"]["val"], "p995": res["crestas"]["respuesta_p995_sanas_train"]}},
    indent=1, ensure_ascii=False, default=float))

# ---------------------------------------------------------------- 4. figuras
uD = res["deficit"]["umbral"]
_, dL, kL, lL, rL = mejorL.split("|"); kL, lL, rL = float(kL), float(lL), float(rL)


def mapas(i):
    In, m, bb = preparar(i); Ic = Mp.limpiar(In, m, bb, "B")
    D = Mp.deficit(Ic, m, sD); R = respuesta(dL, Ic, m)
    return Ic, m, D, R, Mp.grietas(R, Mp.interior(m), kL, lL, rL)[0]


V = part[(part.split == "val") & (part.panel_cod != "M17")]
EJ = []
for t in ("mono", "poly"):
    v = VA[(VA.tipo == t) & (VA.panel != "M17")]
    s = v[v.c4 == 0].sort_values(mejorL).iloc[len(v[v.c4 == 0]) // 2]          # sana típica
    g = v[v.y == 1].sort_values(mejorL).iloc[-3]                              # defectuosa con grieta larga
    a = v[v.y == 1].sort_values(mejorA[t]).iloc[-2]                              # defectuosa con zona apagada grande
    EJ += [(int(s.idx), f"{t} · sana (0 %)"), (int(g.idx), f"{t} · grieta"), (int(a.idx), f"{t} · zona apagada")]

fig, axs = plt.subplots(3, 6, figsize=(15.5, 8.2))
for j, (i, nom) in enumerate(EJ):
    Ic, m, D, R, g = mapas(i)
    axs[0, j].imshow(Ic, cmap="gray", vmin=0, vmax=1.2); axs[0, j].set_title(f"#{i} · {nom}\n{part.prob[i] * 100:.0f} % defecto", loc="left", fontsize=8.5)
    axs[1, j].imshow(Ic, cmap="gray", vmin=0, vmax=1.2)
    axs[1, j].imshow(np.ma.masked_less_equal(D, .05), cmap="magma_r", vmin=0, vmax=.8, alpha=.75)
    axs[1, j].contour(D > uD[tipos[i]], [.5], colors="#00a3ff", linewidths=1)
    axs[1, j].set_title(f"área apagada {(D[Mp.interior(m)] > uD[tipos[i]]).mean() * 100:.1f} %", loc="left", fontsize=8.5)
    axs[2, j].imshow(R, cmap="magma", vmin=0, vmax=np.percentile(R[Mp.interior(m)], 99.8))
    axs[2, j].contour(g, [.5], colors="#12d6b0", linewidths=.9)
    axs[2, j].set_title(f"grietas {Mp.grietas(R, Mp.interior(m), kL, lL, rL)[1]:.0f} px", loc="left", fontsize=8.5)
    for a in axs[:, j]: a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel("celda limpia Ic (B)", fontsize=9)
axs[1, 0].set_ylabel(f"déficit D (σ = {sD:g})\nazul: D > {uD['mono']:g} mono, > {uD['poly']:g} poly", fontsize=9)
axs[2, 0].set_ylabel(f"crestas ({dL})\nverde: grieta", fontsize=9)
fig.subplots_adjust(wspace=.04, hspace=.16)
guardar(fig, os.path.join(FIG, "m1_mapas.png"))

# m2: una poly sana con Sato y con Gabor, y el mismo par con una grieta sintética
i = int(V[(V.tipo == "poly") & (V.clase4 == 0)].sample(1, random_state=11).index[0])
In, m, bb = preparar(i); n = Mp.interior(m)
l = linea(150, 120, 35, 70)
cols = [("Sato σ 1-2", In), ("Gabor λ8", In), ("Sato σ 1-2", In * (1 - .2 * l)), ("Gabor λ8", In * (1 - .2 * l))]
fig, axs = plt.subplots(1, 5, figsize=(15.5, 3.4))
axs[0].imshow(Mp.limpiar(In * (1 - .2 * l), m, bb, "B"), cmap="gray", vmin=0, vmax=1.2)
axs[0].set_title(f"#{i} · poly sana\n+ grieta sintética al 20 %", loc="left", fontsize=8.5)
for a, (det, I) in zip(axs[1:], cols):
    Ic = Mp.limpiar(I, m, bb, "B"); R = respuesta(det, Ic, m)
    _, _, k, lm, r = mejor_det[det].split("|")
    g, Lt, _ = Mp.grietas(R, n, float(k), float(lm), float(r))
    a.imshow(R, cmap="magma", vmin=0, vmax=np.percentile(R[n], 99.8)); a.contour(g, [.5], colors="#12d6b0", linewidths=.9)
    a.set_title(f"{det} · {'con' if I is not In else 'sin'} grieta\nmarcado {Lt:.0f} px", loc="left", fontsize=8.5)
for a in axs: a.set_xticks([]); a.set_yticks([])
fig.subplots_adjust(wspace=.04)
guardar(fig, os.path.join(FIG, "m2_poly_granos.png"))

# m3: AUC de una sola cifra de cada mapa, en entrenamiento (claro) y validación (oscuro), mono y poly
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
filas = [(d, {t: mejor_det[d] for t in ("mono", "poly")}, tL) for d in DETECTORES] + [
    ("Déficit · regla sana ≤ 1 %", mejorA, tA), ("Déficit · sin regla (" + sin_regla.split("|")[2] + ")", {t: sin_regla for t in ("mono", "poly")}, tA)]
tVL = tabla(VA, list(mejor_det.values()))
y = np.arange(len(filas))
for k, t in enumerate(("mono", "poly")):
    for q, (nom_s, alfa) in enumerate((("entrenamiento", .35), ("validación", 1))):
        v = [(tab if q == 0 else (tVL | tV))[cl[t]][t]["auc"] for _, cl, tab in filas]
        yy_ = y + (2 * k + q - 1.5) * .2
        axs[0].barh(yy_, v, .18, color=COL_TIPO[t], alpha=alfa, label=f"{NOMBRE_TIPO[t]} · {nom_s}")
        for a_, b_ in zip(yy_, v):
            axs[0].text(b_ + .004, a_, f"{b_:.2f}", va="center", fontsize=6.5, color=TINTA2)
axs[0].set_yticks(y, [f[0] for f in filas]); axs[0].invert_yaxis(); axs[0].set_xlim(.5, 1)
axs[0].set_xlabel("AUC de una sola cifra del mapa (largo de grietas o área apagada)")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.45, -.16), ncol=4)
axs[0].set_title("Qué tanto separa cada mapa, solo", loc="left")
for d, ls in zip(DETECTORES, ("-", "--", ":", "-.")):
    for t in ("mono", "poly"):
        z = res["sinteticas"]["detectada"][d][t]
        axs[1].plot([100 * c for c in z], [100 * v for v in z.values()], ls, color=COL_TIPO[t], marker="o", ms=4, lw=1.6,
                    label=d if t == "mono" else None)
axs[1].set_xticks([10, 20, 30]); axs[1].set_xlabel("contraste de la grieta sintética (%)"); axs[1].set_ylabel("detectadas (%)"); axs[1].set_ylim(-3, 103)
axs[1].set_title("Grietas sintéticas (azul mono, naranja poly)", loc="left"); axs[1].legend(frameon=False, fontsize=7.5)
guardar(fig, os.path.join(FIG, "m3_barrido_mapas.png"))
print(f"listo en {time.time() - t0:.0f} s")
