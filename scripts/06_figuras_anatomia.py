"""Semana 2 · Figuras rotuladas: qué es un busbar, qué es un dedo, qué es un defecto y por qué se quita la rejilla.

Uso:   python scripts/06_figuras_anatomia.py

Solo usa celdas de ENTRENAMIENTO. ELPV no trae anotaciones por píxel: los rótulos de defectos son lectura visual;
los busbars se detectan con busbar_rows() y los dedos con el perfil por columnas.

Salidas: reports/figures/b0a_anatomia_celda.png, b0b_rejilla_vs_defectos.png, b0c_por_que_quitar_rejilla.png
         reports/metricas/semana2_anatomia.json
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
from matplotlib.patches import Ellipse, Rectangle
from skimage.filters import sato

from datos import cargar_imagenes
from estilo import TINTA, TINTA2, aplicar_estilo, guardar
from preproc import busbar_rows, cell_masks, crack_mask, image_mask, preprocess

aplicar_estilo()
FIG = os.path.join(RAIZ, "reports", "figures")
METAL, DEFECTO, NOCUENTA = "#00a08c", "#e02020", "#8a5a00"   # rejilla normal / defecto / no cuenta en ELPV

part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])
assert (part.split[[923, 6, 15, 282, 2371]] == "train").all()
medidas = {}


def mostrar(ax, k):
    im = ims[k]
    ax.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.8))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rotulo(ax, texto, punto, caja, col, ha="center", fs=8.5):
    """Etiqueta con flecha: `punto` es lo señalado y `caja` dónde va el texto (en píxeles de la celda)."""
    ax.annotate(texto, xy=punto, xytext=caja, ha=ha, va="center", fontsize=fs, color=TINTA, fontweight="bold",
                annotation_clip=False, zorder=6, bbox=dict(boxstyle="round,pad=.3", fc="white", ec=col, lw=1.5),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, shrinkA=0, shrinkB=0))


def caja(ax, x, y, w, h, col, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=2, ls=ls))


# filas reales por inspección visual: busbar_rows() (v1) también devuelve falsos, que se descartan y se anotan
REALES = {923: [51, 150, 248], 15: [51, 150, 248], 6: [51, 150, 248], 282: [80, 223], 2371: [51, 150, 248]}


def busbars(k):
    det = [int(r) for r in busbar_rows(ims[k], image_mask(ims[k], T[tipos[k]]))[1]]
    ok = [r for r in det if any(abs(r - c) <= 6 for c in REALES[k])]
    medidas[f"celda_{k}_busbars"] = {"reales": ok, "falsos_detectados_v1": [r for r in det if r not in ok]}
    assert len(ok) == len(REALES[k])
    return ok


def marcar_busbars(ax, filas, x_texto, fs):
    for j, r in enumerate(filas):
        caja(ax, 3, r - 6, 292, 12, METAL)
        rotulo(ax, f"BUSBAR {j + 1}", (296, r), (x_texto, r), METAL, ha="left", fs=fs)


# ---------- b0a · anatomía de una celda sana ----------
K = 923
Y0, Y1, X0, X1 = 70, 132, 130, 180              # zona entre los busbars 1 y 2
bb = busbars(K)
fig = plt.figure(figsize=(14.6, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.35, 1], wspace=.5, hspace=.42)
a = fig.add_subplot(gs[:, 0]); mostrar(a, K)
a.set_title(f"1. Celda sana completa (mono, #{K})", loc="left", pad=32)
marcar_busbars(a, bb, 316, 8.5)
caja(a, X0, Y0, X1 - X0, Y1 - Y0, TINTA, "--")
rotulo(a, "Aquí hay DEDOS\n(verticales, casi invisibles)", (X0, 101), (65, 101), METAL, fs=8)
rotulo(a, "SILICIO\n(emite luz)", (225, 190), (225, 218), TINTA2, fs=8)
rotulo(a, "ESQUINA CORTADA (normal en mono)", (9, 9), (85, -14), TINTA2, fs=8)
rotulo(a, "DEDO INTERRUMPIDO\n(probable)\nno cuenta", (117, 200), (52, 200), NOCUENTA, fs=8)

# realce: dividir por el fondo y promediar en vertical, que es la dirección de los dedos
f = ims[K].astype(np.float32)
z = f / cv2.GaussianBlur(f, (0, 0), 5)
zr = cv2.GaussianBlur(z, (0, 0), sigmaX=.01, sigmaY=8)
prof = z[Y0:Y1].mean(0) * 100
F = np.abs(np.fft.rfft((prof[30:270] - prof[30:270].mean()) * np.hanning(240), 4096))
fr = np.fft.rfftfreq(4096)
paso = float(1 / fr[np.argmax(F * ((fr > 1 / 7) & (fr < .45)))])
xs, p = np.arange(X0, X1), prof[X0:X1]
val = [i for i in range(1, len(p) - 1) if p[i] < p[i - 1] and p[i] <= p[i + 1]
       and min(p[max(0, i - 2):i].max(), p[i + 1:i + 3].max()) - p[i] > .2]   # valles de al menos 0,2 %
medidas["celda_923_paso_dedos_px"] = round(paso, 2)
medidas["celda_923_contraste_dedos_pct"] = round(float(np.percentile(p, 90) - np.percentile(p, 10)), 2)

a = fig.add_subplot(gs[0, 1])
a.imshow(zr[Y0:Y1, X0:X1], cmap="gray", vmin=.982, vmax=1.018, extent=(X0 - .5, X1 - .5, Y1, Y0), aspect="auto",
         interpolation="nearest")
a.set_xticks([]); a.set_yticks([])
for i in val:
    a.plot([X0 + i] * 2, [Y0 - 1, Y0 - 4.5], color=METAL, lw=2.4, clip_on=False)
a.set_xlim(X0 - .5, X1 - .5); a.set_ylim(Y1, Y0)
a.set_title("2. Recuadro realzado: cada marca verde es un DEDO", loc="left", pad=18)
a = fig.add_subplot(gs[1, 1])
a.plot(xs, p, color=TINTA2, lw=1.6); a.plot(xs[val], p[val], "v", color=METAL, ms=7)
a.set_xlim(X0 - .5, X1 - .5); a.set_xlabel("Columna (px)"); a.set_ylabel("Brillo (% del fondo)")
a.set_title("3. Brillo por columna: cada valle es un dedo (uno cada " + f"{paso:.1f}".replace(".", ",") + " px)", loc="left")
guardar(fig, os.path.join(FIG, "b0a_anatomia_celda.png"))

# ---------- b0b · rejilla normal frente a defectos ----------
fig, axs = plt.subplots(2, 2, figsize=(12.6, 11.4))
for ax, k, t in [(axs[0, 0], 15, "Grieta, zona apagada y dedo interrumpido"), (axs[0, 1], 6, "Zona apagada grande y grieta"),
                 (axs[1, 0], 282, "Celda de 2 busbars con grieta"), (axs[1, 1], 2371, "Celda sana con dedos interrumpidos")]:
    mostrar(ax, k)
    ax.set_title(f"{t}\n{tipos[k]} · #{k} · etiqueta {part.prob[k] * 100:.0f} %", loc="left")
a = axs[0, 0]
marcar_busbars(a, busbars(15), 310, 8)
a.add_patch(Ellipse((246, 99), 24, 134, angle=-45, fill=False, ec=DEFECTO, lw=2))
rotulo(a, "GRIETA", (216, 88), (140, 100), DEFECTO)
caja(a, 22, 4, 258, 27, DEFECTO)
rotulo(a, "ZONA APAGADA", (86, 16), (170, 17), DEFECTO)
caja(a, 178, 256, 16, 41, NOCUENTA, "--")
rotulo(a, "DEDO INTERRUMPIDO\n(probable): no cuenta", (178, 278), (95, 278), NOCUENTA, fs=8)
a = axs[0, 1]
marcar_busbars(a, busbars(6), 310, 8)
caja(a, 8, 88, 250, 78, DEFECTO)
rotulo(a, "ZONA APAGADA", (110, 166), (110, 195), DEFECTO)
rotulo(a, "GRIETA", (255, 78), (200, 25), DEFECTO)
a = axs[1, 0]
marcar_busbars(a, busbars(282), 310, 8)
rotulo(a, "GRIETA", (150, 128), (110, 180), DEFECTO)
rotulo(a, "TEXTURA DE GRANO (normal en poly)", (190, 55), (130, 22), TINTA2, fs=8)
a = axs[1, 1]
marcar_busbars(a, busbars(2371), 310, 8)
caja(a, 155, 160, 12, 70, NOCUENTA, "--"); caja(a, 49, 70, 12, 76, NOCUENTA, "--")
rotulo(a, "DEDO INTERRUMPIDO\n(probable): no cuenta", (167, 195), (226, 205), NOCUENTA, fs=8)
rotulo(a, "DEDO INTERRUMPIDO\n(probable): no cuenta", (61, 105), (130, 95), NOCUENTA, fs=8)
rotulo(a, "DEDOS: rayado vertical tenue", (262, 225), (190, 273), METAL, fs=8)
fig.subplots_adjust(wspace=.3, hspace=.16, bottom=.06)
for c, ls, t, x in [(METAL, "-", "Metal normal (se quita de la imagen)", .13), (DEFECTO, "-", "Defecto (se quiere encontrar)", .42),
                    (NOCUENTA, "--", "Se parece a un defecto, pero no cuenta", .66)]:
    fig.add_artist(plt.Line2D([x, x + .03], [.035] * 2, color=c, ls=ls, lw=2.5))
    fig.text(x + .037, .035, t, va="center", fontsize=9.5, color=TINTA)
guardar(fig, os.path.join(FIG, "b0b_rejilla_vs_defectos.png"), dpi=130)

# ---------- b0c · qué pasa si NO se quita la rejilla ----------
fig, axs = plt.subplots(2, 4, figsize=(13, 6.9))
for i, (k, nom) in enumerate([(923, "Celda SANA"), (15, "Celda con GRIETA")]):
    m = image_mask(ims[k], T[tipos[k]]); pp = preprocess(ims[k], m)
    con = np.clip(pp["norm"], 0, 1.3).astype(np.float32); con[~m] = np.median(con[m])
    r0 = sato(con, sigmas=[1, 2], black_ridges=True); r0[~pp["inner"]] = 0
    c0, c1 = crack_mask({"ridge": r0}, pp["inner"]), crack_mask(pp, pp["inner"])
    medidas[f"celda_{k}_px_marcados_con_rejilla"] = int(c0.sum())
    medidas[f"celda_{k}_px_marcados_sin_rejilla"] = int(c1.sum())
    pasos = [(con, None, "1. Con rejilla"), (con, c0, "2. Detector sobre la 1"),
             (pp["nobus"], None, "3. Sin rejilla"), (pp["nobus"], c1, "4. Detector sobre la 3")]
    for j, (im, c, tt) in enumerate(pasos):
        ax = axs[i, j]; ax.imshow(im, cmap="gray", vmin=.3, vmax=1.15); ax.set_xticks([]); ax.set_yticks([])
        if c is not None:
            o = np.zeros((*c.shape, 4)); o[c] = (.9, .1, .1, 1); ax.imshow(o, interpolation="nearest")
            ax.set_xlabel(f"{int(c.sum())} px marcados como grieta", fontsize=9)
        if i == 0:
            ax.set_title(tt, loc="left")
    axs[i, 0].set_ylabel(f"{nom}\n#{k}", fontsize=9.5)
fig.subplots_adjust(wspace=.05, hspace=.16)
guardar(fig, os.path.join(FIG, "b0c_por_que_quitar_rejilla.png"))

with open(os.path.join(RAIZ, "reports", "metricas", "semana2_anatomia.json"), "w", encoding="utf-8") as fh:
    json.dump(medidas, fh, indent=2, ensure_ascii=False)
print(json.dumps(medidas, indent=1))
