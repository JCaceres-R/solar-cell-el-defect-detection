"""Semana 2 · Revisión de §4.3: ¿qué le hace el cierre de dedos a la respuesta del filtro de crestas (Sato)?

Uso:   python scripts/09_grietas_verticales.py      (~3 min; la prueba NO se carga)

Todo parte de la salida de B (cierre de busbars). Se compara sin cierre de dedos frente a 1x3, 1x5 y 1x7:
  1. Grietas sintéticas en 80 celdas sanas de VALIDACIÓN: respuesta de Sato sobre la grieta y % de grietas que se detectan.
  2. Trazos reales de VALIDACIÓN: los que Sato marca sin cierre de dedos (alargados, de 25 px o más), por orientación.
     Se dejan fuera M17 (sus pestañas laterales dan trazos verticales falsos) y los trazos a menos de 14 px del borde lateral
     de la máscara, que son el escalón del borde. ELPV no dice si un trazo es grieta o dedo interrumpido, así que se separan por la etiqueta de su celda.
  3. Figura b4 con celdas de ENTRENAMIENTO, antes y después.

Salidas: reports/figures/b4_grietas_verticales.png, b4b_sato_dedos.png y reports/metricas/semana2_grietas_verticales.json
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
from skimage import measure
from skimage.filters import sato

import busbars as B
from datos import cargar_imagenes
from estilo import TINTA2, aplicar_estilo, guardar
from preproc import cell_masks, crack_mask, image_mask

aplicar_estilo()
FIG, MET = os.path.join(RAIZ, "reports", "figures"), os.path.join(RAIZ, "reports", "metricas")
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])
ANCHOS, ANG = [1, 3, 5, 7], [0, 15, 30, 45, 60, 75, 90]
COL_A = {1: "#8b8a85", 3: "#b39ae8", 5: "#7a5bd0", 7: "#3c2596"}


def preparar(i):
    m = image_mask(ims[i], T[tipos[i]]); f = ims[i].astype(np.float32)
    In = f / (np.percentile(f[m], 90) + 1e-6)
    return In, m, B.detectar(In, m)


def respuesta(I, m):
    """Respuesta de Sato y máscara de grietas, con los mismos parámetros que la línea base."""
    n = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    f = np.clip(I, 0, 1.3).astype(np.float32).copy(); f[~m] = np.median(f[m])
    r = sato(f, sigmas=[1, 2], black_ridges=True); r[~n] = 0
    return r, crack_mask({"ridge": r}, n), n


yy, xx = np.mgrid[:300, :300].astype(np.float32)


def linea(cx, cy, ang, largo, sigma):
    a = np.deg2rad(ang); dx, dy = np.cos(a), -np.sin(a)
    t, n = (xx - cx) * dx + (yy - cy) * dy, -(xx - cx) * dy + (yy - cy) * dx
    return np.exp(-n ** 2 / (2 * sigma ** 2)) * (np.abs(t) < largo / 2)


# ---------------------------------------------------------------- 1. sintéticas
def sintetica(i):
    rng = np.random.default_rng(int(i)); In, m, bb = preparar(i); out = []
    if len(bb) < 2:
        return out
    for ang in ANG:
        # 60 px, centrada entre dos busbars: una grieta vertical cabe entera fuera de las bandas
        l = linea(rng.uniform(100, 200), (bb[0]["fila"] + bb[1]["fila"]) / 2, ang, 60, 1.0)
        Ib = B.cierre(In * (1 - .2 * l), m, bb); nucleo = l > .6
        r0 = None
        for a in ANCHOS:
            r, c, n = respuesta(B.quitar_dedos(Ib, a), m); w = nucleo & n
            v = float(r[w].mean()); r0 = v if r0 is None else r0
            out.append({"tipo": tipos[i], "ang": ang, "ancho": a, "sato_rel": v / r0, "detectada": bool(c[w].mean() >= .5)})
    return out


ok = part[(part.split == "val") & (part.clase4 == 0) & (part.panel_cod != "M17")]
base = list(ok[ok.tipo == "mono"].sample(40, random_state=3).index) + list(ok[ok.tipo == "poly"].sample(40, random_state=3).index)
S = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(sintetica)(i) for i in base) for x in r])
res = {"sinteticas_n": int(len(S) / len(ANG) / len(ANCHOS)), "sinteticas": {}}
for t in ["todas", "mono", "poly"]:
    g = (S if t == "todas" else S[S.tipo == t]).groupby(["ancho", "ang"])[["sato_rel", "detectada"]].mean()
    res["sinteticas"][t] = {str(a): {"sato_rel": g.sato_rel[a].round(4).to_dict(), "detectada": g.detectada[a].round(4).to_dict()} for a in ANCHOS}


# ---------------------------------------------------------------- 2. trazos reales
def trazos(i):
    In, m, bb = preparar(i)
    if len(bb) < 2:
        return []
    Ib = B.cierre(In, m, bb)
    R = {a: respuesta(B.quitar_dedos(Ib, a), m) for a in ANCHOS}
    out = []
    for p in measure.regionprops(measure.label(R[1][1])):
        if p.axis_major_length < 25 or p.eccentricity < .95:
            continue
        inc = 90 - abs(np.degrees(p.orientation))          # 0 horizontal, 90 vertical
        y, x = p.coords[:, 0], p.coords[:, 1]
        lados = np.where(m.any(0))[0]
        if p.centroid[1] < lados[0] + 14 or p.centroid[1] > lados[-1] - 14:
            continue
        # se toma la cresta central (25 % de píxeles de mayor respuesta), que es donde está la línea
        k = R[1][0][y, x] >= np.percentile(R[1][0][y, x], 75)
        d = {"idx": i, "tipo": tipos[i], "clase4": int(part.clase4[i]), "inc": float(inc), "largo": float(p.axis_major_length),
             "cy": float(p.centroid[0]), "cx": float(p.centroid[1])}
        for a in ANCHOS:
            d[f"sato_{a}"] = float(R[a][0][y, x][k].mean() / R[1][0][y, x][k].mean())
            d[f"sigue_{a}"] = bool(R[a][1][y, x].mean() >= .5)
        out.append(d)
    return out


def juntar(idx):
    return pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(trazos)(int(i)) for i in idx) for x in r])


V = juntar(np.where((part.split == "val").values)[0])
V["orient"] = pd.cut(V.inc, [-1, 30, 60, 91], labels=["horizontal", "inclinado", "vertical"])
res["reales"] = {}
for t in ["todas", "mono", "poly"]:
    q = V if t == "todas" else V[V.tipo == t]
    res["reales"][t] = {}
    for e, s in [("todas", q), ("sanas_0", q[q.clase4 == 0]), ("defectuosas_100", q[q.clase4 == 3])]:
        res["reales"][t][e] = {o: {"n": int(len(z)), **{f"sato_{a}": float(z[f"sato_{a}"].median()) if len(z) else None for a in ANCHOS},
                                   **{f"sigue_{a}": float(z[f"sigue_{a}"].mean()) if len(z) else None for a in ANCHOS}}
                               for o, z in s.groupby("orient", observed=False)}
json.dump(res, open(os.path.join(MET, "semana2_grietas_verticales.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))

# ---------------------------------------------------------------- 3. figuras
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for a in ANCHOS[1:]:
    d = res["sinteticas"]["todas"][str(a)]["sato_rel"]
    axs[0].plot(list(d), [100 * v for v in d.values()], "o-", color=COL_A[a], lw=2, ms=5); axs[0].text(92, 100 * d[90], f"1×{a}", fontsize=8, color=TINTA2, va="center")
for a in ANCHOS:
    d = res["sinteticas"]["todas"][str(a)]["detectada"]
    axs[1].plot(list(d), [100 * v for v in d.values()], "o-", color=COL_A[a], lw=2, ms=5, label="sin cierre" if a == 1 else f"1×{a}")
axs[0].set_title("Sintéticas: respuesta de Sato que queda (%)", loc="left"); axs[1].set_title("Sintéticas: grietas detectadas (%)", loc="left")
for a in axs[:2]:
    a.set_xticks(ANG); a.set_ylim(0, 108); a.set_xlim(-3, 102); a.set_xlabel("Ángulo (0° horizontal · 90° vertical)")
axs[1].legend(frameon=False, fontsize=8, loc="lower left")
ors = ["horizontal", "inclinado", "vertical"]
for j, a in enumerate(ANCHOS[1:]):
    y = [100 * res["reales"]["todas"]["todas"][o][f"sigue_{a}"] for o in ors]
    axs[2].bar(np.arange(3) + (j - 1) * .27, y, .24, color=COL_A[a], label=f"1×{a}")
    for q, v in enumerate(y): axs[2].text(q + (j - 1) * .27, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=TINTA2)
axs[2].set_xticks(range(3), [f"{o}\n(n = {res['reales']['todas']['todas'][o]['n']})" for o in ors]); axs[2].set_ylim(0, 108)
axs[2].set_title("Reales: trazos que Sato sigue marcando (%)", loc="left"); axs[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2))
fig.subplots_adjust(wspace=.22)
guardar(fig, os.path.join(FIG, "b4b_sato_dedos.png"))

Tr = juntar(np.where((part.split == "train").values & (part.clase4 == 3).values)[0])
Tr = Tr[(Tr.inc >= 70) & (Tr.largo >= 35)].sort_values("largo", ascending=False)
if "SALIDA_TMP" in os.environ:
    Tr.to_csv(os.path.join(os.environ["SALIDA_TMP"], "_candidatos_tmp.csv"), index=False)
# elegidos a ojo en entrenamiento: una grieta real muy inclinada y un dedo interrumpido probable
CASOS = [[869, 158, 240, "#869 · grieta"], [2371, 161, 195, "#2371 · dedo interrumpido (probable)"]]
fig, axs = plt.subplots(2, 4 * len(CASOS) + 1, figsize=(15.6, 4.1), gridspec_kw={"width_ratios": [1] * 4 + [.12] + [1] * 4})
for a in axs[:, 4]: a.axis("off")
for c, (i, cx, cy, nom) in enumerate(CASOS):
    In, m, bb = preparar(i); Ib = B.cierre(In, m, bb)
    z = (slice(max(0, cy - 45), min(300, cy + 45)), slice(max(0, cx - 45), min(300, cx + 45)))
    vm = respuesta(Ib, m)[0][z].max()
    for j, a in enumerate(ANCHOS):
        I = B.quitar_dedos(Ib, a); r, k, _ = respuesta(I, m)
        p, q = axs[0, 5 * c + j], axs[1, 5 * c + j]
        p.imshow(I[z], cmap="gray", vmin=np.percentile(Ib[z], 1), vmax=np.percentile(Ib[z], 99.5))
        q.imshow(r[z], cmap="magma", vmin=0, vmax=vm); q.contour(k[z], [.5], colors="#12b39b", linewidths=1)
        p.set_title((nom + "\n" if j == 0 else "\n") + ("sin cierre" if a == 1 else f"cierre 1×{a}"), loc="left", fontsize=9)
        for x in (p, q): x.set_xticks([]); x.set_yticks([])
axs[0, 0].set_ylabel("imagen", fontsize=8.5); axs[1, 0].set_ylabel("Sato · verde: marcado", fontsize=8.5)
fig.subplots_adjust(wspace=.05, hspace=.05)
guardar(fig, os.path.join(FIG, "b4_grietas_verticales.png"), dpi=130)
