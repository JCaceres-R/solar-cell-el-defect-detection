"""Semana 2 · Busbars: detección v2, notch (A) frente a cierre (B), y dedos. Figuras b1-b6 y métricas.

Uso:   python scripts/07_comparar_busbars.py        (~4 min; la prueba NO se carga en ningún momento)

  0. Detección: v1 (línea base) frente a v2, contra la rejilla de cada panel (lectura de su imagen mediana).
  1. Barrido de parámetros en ENTRENAMIENTO (200 celdas sanas).
  2. Criterios 1, 2 y 4 de la tabla 3 en VALIDACIÓN, mono y poly por separado.
  3. Grietas sintéticas (apoyo, no está en la tabla 3): qué fracción de su contraste conserva cada método según el ángulo.
  4. Dedos: cierre horizontal de 1, 3, 5 y 7 px; contraste de los dedos que queda, grietas que sobreviven y falsas crestas.

Salidas: reports/figures/b1..b6, reports/metricas/semana2_busbars.json y semana2_por_celda_val.csv
"""
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.filters import sato

import busbars as B
from datos import cargar_imagenes
from estilo import COL_TIPO, GRIS, NOMBRE_TIPO, TINTA, TINTA2, aplicar_estilo, guardar
from preproc import busbar_rows, cell_masks, crack_mask, image_mask

aplicar_estilo()
FIG, MET = os.path.join(RAIZ, "reports", "figures"), os.path.join(RAIZ, "reports", "metricas")
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
T = cell_masks(ims[(part.split == "train").values], tipos[(part.split == "train").values])
usar = np.where(part.split.isin(["train", "val"]).values)[0]
res = {"parametros": {"sigma_n": B.SIGMA_N, "K": B.K_MAX, "h_cierre": B.H_CIERRE, "ancho_dedo": B.ANCHO_DEDO}}

# Rejilla de cada panel, leída en la imagen mediana del panel (las manchas de cada celda se cancelan). Es la referencia
# para medir la detección, porque ELPV no anota los busbars.
PANELES_2 = {"P01", "P04", "P05", "P11", "P20", "P24"}
PANELES_0 = {"M17"}                                  # contacto trasero: no hay metal en la cara frontal
METODOS = ["original", "A-global", "A-filas", "B"]
COL_M = {"original": "#8b8a85", "A-global": "#7a5bd0", "A-filas": "#b97a00", "B": "#00846f"}


def rejilla(panel):
    return [] if panel in PANELES_0 else [80, 222] if panel in PANELES_2 else [51, 150, 248]


def acierta(filas, ref, tol=9):
    return len(filas) == len(ref) and all(abs(a - b) <= tol for a, b in zip(filas, ref))


def preparar(i):
    m = image_mask(ims[i], T[tipos[i]])
    f = ims[i].astype(np.float32)
    In = f / (np.percentile(f[m], 90) + 1e-6)
    return In, m, B.detectar(In, m)


def salidas(In, m, bb):
    return {"original": B.rellenar_fuera(In, m), "A-global": B.notch(In, m, bb), "A-filas": B.notch_filas(In, m, bb),
            "B": B.cierre(In, m, bb)}


def interior(m):
    return cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)


def crestas(I, m):
    """Píxeles que el filtro de crestas marca como grieta (mismos parámetros que la línea base)."""
    n = interior(m)
    f = np.clip(I, 0, 1.3).astype(np.float32).copy(); f[~m] = np.median(f[m])
    r = sato(f, sigmas=[1, 2], black_ridges=True); r[~n] = 0
    return crack_mask({"ridge": r}, n)


yy, xx = np.mgrid[:300, :300].astype(np.float32)


def linea(cx, cy, ang, largo, sigma):
    """Grieta sintética: línea gaussiana. ang = 0 horizontal (como los busbars), 90 vertical (como los dedos)."""
    a = np.deg2rad(ang); dx, dy = np.cos(a), -np.sin(a)
    t, n = (xx - cx) * dx + (yy - cy) * dy, -(xx - cx) * dy + (yy - cy) * dx
    return np.exp(-n ** 2 / (2 * sigma ** 2)) * (np.abs(t) < largo / 2)


# ---------------------------------------------------------------- 0. detección
def det(i):
    In, m, bb = preparar(i)
    ref = rejilla(part.panel_cod[i])
    v1 = [int(r) for r in busbar_rows(ims[i], m)[1]]
    v2 = [b["fila"] for b in bb]
    cerca = lambda f: any(abs(f - r) <= 9 for r in ref)
    return {"idx": i, "split": part.split[i], "tipo": tipos[i], "panel": part.panel_cod[i], "n_ref": len(ref), "n_v1": len(v1),
            "n_v2": len(v2), "ok_v1": acierta(v1, ref), "ok_v2": acierta(v2, ref),
            "falsos_v1": sum(not cerca(f) for f in v1), "falsos_v2": sum(not cerca(f) for f in v2),
            "inferidos": sum(b.get("inferido", False) for b in bb),
            "grosor": float(np.mean([b["grosor"] for b in bb])) if bb else np.nan, "prob": part.prob[i]}


D = pd.DataFrame(Parallel(n_jobs=-1)(delayed(det)(i) for i in usar))
res["deteccion"] = {}
for s in ["train", "val"]:
    for t in ["mono", "poly", "todas"]:
        q = D[(D.split == s) & ((D.tipo == t) | (t == "todas"))]
        res["deteccion"][f"{s}_{t}"] = {"n": len(q), "v1_exacta": q.ok_v1.mean(), "v2_exacta": q.ok_v2.mean(),
                                        "v1_falsos_por_celda": q.falsos_v1.mean(), "v2_falsos_por_celda": q.falsos_v2.mean()}
res["deteccion"]["fallos_v2"] = D[~D.ok_v2][["idx", "split", "panel", "prob", "n_v2"]].to_dict("records")
res["deteccion"]["celdas_con_banda_inferida"] = D[D.inferidos > 0].idx.tolist()
res["deteccion"]["sin_rejilla_val"] = int((D[D.split == "val"].n_ref == 0).sum())
res["deteccion"]["grosor_mediano_px"] = D.groupby("tipo").grosor.median().to_dict()

# ---------------------------------------------------------------- 1. barrido en entrenamiento (celdas sanas)
SIG, KS, HS = [.5, .75, 1, 1.5, 2, 3], [5, 10, 20, 24, 40], [9, 13, 17, 21, 27, 33]
san = part[(part.split == "train") & (part.clase4 == 0)]
ajuste = list(san[san.tipo == "mono"].sample(80, random_state=1).index) + list(san[san.tipo == "poly"].sample(120, random_state=1).index)


def barrer(i):
    In, m, bb = preparar(i); out = []
    if not bb:
        return out
    f = B.rellenar_fuera(In, m)
    for s in SIG:
        for K in (KS if s == 1 else [B.K_MAX]):
            g = B.notch(In, m, bb, s, K)
            out.append(("A-global", s, K, B.residual(g, m, bb)[0], B.residual(g, m, bb, tramo=12)[0], B.cambio_fuera(g, f, m, bb)))
            g = B.notch_filas(In, m, bb, s, K)
            out.append(("A-filas", s, K, B.residual(g, m, bb)[0], B.residual(g, m, bb, tramo=12)[0], B.cambio_fuera(g, f, m, bb)))
    for h in HS:
        g = B.cierre(In, m, bb, h)
        out.append(("B", h, 0, B.residual(g, m, bb)[0], B.residual(g, m, bb, tramo=12)[0], B.cambio_fuera(g, f, m, bb)))
    return out


R = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(barrer)(i) for i in ajuste) for x in r],
                 columns=["metodo", "p", "K", "c1", "c1_tramos", "c2"])
res["barrido_train"] = R.groupby(["metodo", "p", "K"])[["c1", "c1_tramos", "c2"]].mean().reset_index().to_dict("records")


# ---------------------------------------------------------------- 2. criterios en validación
def medir(i):
    In, m, bb = preparar(i); out = []
    if not bb:
        return out
    S = salidas(In, m, bb)
    for k, I in S.items():
        c1, s1 = B.residual(I, m, bb)
        out.append({"idx": i, "tipo": tipos[i], "defectuosa": int(part.defectuosa[i]), "n_busbars": len(bb), "metodo": k,
                    "c1": c1, "c1_signo": s1, "c1_tramos": B.residual(I, m, bb, tramo=12)[0],
                    "c2": B.cambio_fuera(I, S["original"], m, bb),
                    "px_cresta": int(crestas(I, m).sum())})
    return out


val = np.where((part.split == "val").values)[0]
V = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(medir)(i) for i in val) for x in r])
V.round(3).to_csv(os.path.join(MET, "semana2_por_celda_val.csv"), index=False)


def tabla(q):
    g = q.groupby("metodo")[["c1", "c1_signo", "c1_tramos", "c2"]].mean()
    sanas = q[q.defectuosa == 0].groupby("metodo").px_cresta
    return {k: {**g.loc[k].to_dict(), "px_cresta_sanas_mediana": float(sanas.median()[k]),
                "c1_p90": float(q[q.metodo == k].c1.quantile(.9))} for k in METODOS}


res["criterios_val"] = {"n_celdas": int(V.idx.nunique()), "n_por_tipo": V.groupby("tipo").idx.nunique().to_dict(),
                        "todas": tabla(V), "mono": tabla(V[V.tipo == "mono"]), "poly": tabla(V[V.tipo == "poly"]),
                        "dos_busbars": tabla(V[V.n_busbars == 2]), "tres_busbars": tabla(V[V.n_busbars == 3])}

# criterio 4: un solo hilo, mediana de 120 celdas de validación con rejilla, H(u,v) ya calculada para cada periodo
cv2.setNumThreads(1)
tm = {k: [] for k in ["deteccion", "A-global", "A-filas", "B"]}
for i in V.idx.unique()[::3]:
    In, m, bb = preparar(i); B.notch(In, m, bb)
    for k, fn in [("deteccion", lambda: B.detectar(In, m)), ("A-global", lambda: B.notch(In, m, bb)),
                  ("A-filas", lambda: B.notch_filas(In, m, bb)), ("B", lambda: B.cierre(In, m, bb))]:
        t0 = time.perf_counter(); fn(); tm[k].append(1000 * (time.perf_counter() - t0))
res["tiempo_ms"] = {k: float(np.median(v)) for k, v in tm.items()}; res["tiempo_n"] = len(tm["B"])

# ---------------------------------------------------------------- 3. grietas sintéticas (apoyo)
ANG = [0, 5, 10, 15, 30, 45, 60, 75, 90]
sv = part[(part.split == "val") & (part.clase4 == 0) & part.index.isin(V.idx.unique())]
base = list(sv[sv.tipo == "mono"].sample(40, random_state=3).index) + list(sv[sv.tipo == "poly"].sample(40, random_state=3).index)
res["grietas_sinteticas_n"] = len(base)


def grietas(i):
    rng = np.random.default_rng(int(i)); In, m, bb = preparar(i); out = []
    if len(bb) < 2:
        return out
    ban = B.filas_banda(bb)[:, None]
    for ang in ANG:
        # fuera de las bandas: a media distancia entre dos busbars. Cruzando: centrada en un busbar.
        for zona, cy in [("fuera", (bb[0]["fila"] + bb[1]["fila"]) / 2 + rng.uniform(-12, 12)), ("cruza", bb[0]["fila"])]:
            if zona == "cruza" and ang < 15:
                continue
            l = linea(rng.uniform(100, 200), cy, ang, 70, 1.0); Ic = In * (1 - .2 * l)
            S0, S1 = salidas(In, m, bb), salidas(Ic, m, bb)
            w = (l > .6) & m & (~ban if zona == "fuera" else ban)
            # referencia: el contraste que la grieta tendría sobre silicio sano
            ref = (.2 * l * np.median(In[m]))[w].mean() if zona == "cruza" else (In - Ic)[w].mean()
            for k in METODOS:
                out.append((tipos[i], zona, ang, k, float((S0[k] - S1[k])[w].mean() / ref)))
            if zona == "fuera":
                for s in SIG:
                    out.append((tipos[i], "sigma", ang, f"{s}", float((B.notch(In, m, bb, s) - B.notch(Ic, m, bb, s))[w].mean() / ref)))
                for a in [1, 3, 5, 7]:
                    out.append((tipos[i], "dedos", ang, f"{a}", float((B.quitar_dedos(In, a) - B.quitar_dedos(Ic, a))[w].mean() / ref)))
    return out


G = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(grietas)(i) for i in base) for x in r],
                 columns=["tipo", "zona", "ang", "metodo", "ret"])
for z in ["fuera", "cruza", "sigma", "dedos"]:
    res[f"retencion_{z}"] = {k: {int(a): float(v) for a, v in q.groupby("ang").ret.mean().items()} for k, q in G[G.zona == z].groupby("metodo")}


# ---------------------------------------------------------------- 4. dedos en validación (sobre la salida de B)
# Solo celdas con rejilla: M17 es de contacto trasero (no tiene dedos al frente) y sus pestañas laterales dan falsas crestas.
def dedos(i):
    In, m, bb = preparar(i); out = []
    if len(bb) < 2:
        return out
    Ib = B.cierre(In, m, bb)
    y0, y1 = bb[0]["y1"] + 8, bb[1]["y0"] - 8
    for a in [1, 3, 5, 7]:
        I = B.quitar_dedos(Ib, a)
        p = I[y0:y1, 40:260].mean(0); p = p / cv2.GaussianBlur(p.reshape(1, -1), (0, 0), 5).ravel()
        Fq = np.abs(np.fft.rfft((p - p.mean()) * np.hanning(220), 2048)) * 4 / 220
        fr = np.fft.rfftfreq(2048); s = (fr > 1 / 5.6) & (fr < 1 / 3.2)
        out.append({"idx": i, "tipo": tipos[i], "sana": part.clase4[i] == 0, "defectuosa": int(part.defectuosa[i]), "ancho": a,
                    "contraste_dedos_pct": 100 * float(Fq[s].max()), "paso_px": float(1 / fr[np.argmax(Fq * s)]),
                    "px_cresta": int(crestas(I, m).sum())})
    return out


E = pd.DataFrame([x for r in Parallel(n_jobs=-1)(delayed(dedos)(i) for i in val) for x in r])
res["dedos_val"] = {t: {str(a): {"contraste_pct": float(q.contraste_dedos_pct.median()), "px_cresta_sanas": float(q[q.sana].px_cresta.mean()),
                                 "px_cresta_defectuosas": float(q[q.defectuosa == 1].px_cresta.mean())}
                        for a, q in E[(E.tipo == t) | (t == "todas")].groupby("ancho")} for t in ["mono", "poly", "todas"]}

res["dedos_val_n_sanas"] = E[(E.ancho == 1) & E.sana].groupby("tipo").size().to_dict()
res["dedos_val_n"] = int((E.ancho == 1).sum())
json.dump(res, open(os.path.join(MET, "semana2_busbars.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)


# ================================================================ figuras
def sin_ejes(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)


# b1 · espectro y H(u, v)
fig, axs = plt.subplots(2, 4, figsize=(12.4, 6.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
for j, (i, nom) in enumerate([(923, "3 busbars (mono)"), (282, "2 busbars (poly)")]):
    In, m, bb = preparar(i); P = B.periodo(bb); f = B.rellenar_fuera(In, m)
    E0 = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))))
    Hf = B.filtro_notch(P); z = slice(150 - 45, 150 + 46)
    for a, im, t, cm in [(axs[j, 0], f, f"Celda #{i} · {nom}", "gray"), (axs[j, 1], E0[z, z], "Espectro |F| (log, centro)", "magma"),
                         (axs[j, 2], Hf[z, z], f"H(u,v): P = {P:.0f} px, v0 = {300 / P:.1f}", "gray")]:
        a.imshow(im, cmap=cm); a.set_title(t, loc="left", fontsize=9); sin_ejes(a)
    a = axs[j, 3]; k = np.arange(0, 61)
    a.plot(k, np.abs(np.fft.fft2(f - f.mean()))[:61, 0] / 1e3, color=TINTA2, lw=1.4)
    for q in range(1, B.K_MAX + 1):
        if q * 300 / P < 60: a.axvline(q * 300 / P, color="#80393d", lw=.7, ls=":")
    a.set_title("|F| sobre el eje vertical (u = 0)", loc="left", fontsize=9); a.set_xlabel("v (ciclos por imagen)"); a.set_ylabel("magnitud / 1000")
fig.subplots_adjust(wspace=.16, hspace=.32)
guardar(fig, os.path.join(FIG, "b1_espectro_notch.png"))

# b2 · antes / después
CEL = [(923, "mono sana"), (15, "mono con grieta"), (282, "poly con grieta, 2 busbars"), (2371, "poly sana")]
fig, axs = plt.subplots(4, 7, figsize=(15, 8.9))
for r, (i, nom) in enumerate(CEL):
    In, m, bb = preparar(i); S = salidas(In, m, bb)
    for c, k in enumerate(METODOS):
        a = axs[r, c]; a.imshow(np.where(m, S[k], np.nan), cmap="gray", vmin=.3, vmax=1.15); sin_ejes(a)
        if r == 0: a.set_title({"original": "Antes (normalizada)"}.get(k, k), loc="left")
    for c, k in enumerate(METODOS[1:]):
        a = axs[r, 4 + c]; im = a.imshow(np.where(m, S[k] - S["original"], np.nan), cmap="RdBu_r", vmin=-.3, vmax=.3); sin_ejes(a)
        if r == 0: a.set_title(f"Cambio: {k}", loc="left")
    axs[r, 0].set_ylabel(f"{nom}\n#{i}", fontsize=8.5)
fig.subplots_adjust(wspace=.05, hspace=.06, right=.93)
cb = fig.colorbar(im, cax=fig.add_axes([.94, .35, .008, .3])); cb.set_label("rojo: aclara · azul: oscurece", fontsize=8); cb.outline.set_visible(False)
guardar(fig, os.path.join(FIG, "b2_antes_despues.png"), dpi=130)

# b3 · perfil por fila
fig, axs = plt.subplots(1, 2, figsize=(12, 3.3), sharey=True)
for a, (i, nom) in zip(axs, [(923, "mono sana, 3 busbars"), (282, "poly con grieta, 2 busbars")]):
    In, m, bb = preparar(i); S = salidas(In, m, bb)
    for b in bb: a.axvspan(b["y0"] - B.MARGEN, b["y1"] + B.MARGEN, color="#efe9e9", lw=0)
    for k in METODOS:
        a.plot(range(20, 280), [S[k][y, m[y]].mean() if m[y].sum() > 100 else np.nan for y in range(20, 280)], color=COL_M[k],
               lw=1.1 if k == "original" else 1.7, label=k, ls="--" if k == "A-filas" else "-")
    a.set_title(f"#{i} · {nom}", loc="left"); a.set_xlabel("Fila (px)")
axs[0].set_ylabel("Brillo medio de la fila (1 = sano)"); axs[0].legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.02))
guardar(fig, os.path.join(FIG, "b3_perfil_filas.png"))

# b4 · dedos: qué se gana y qué se pierde
fig, axs = plt.subplots(1, 3, figsize=(13, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
an = [1, 3, 5, 7]; etq = ["sin\ncierre", "1×3", "1×5", "1×7"]
for t in ["mono", "poly"]:
    for a, campo in [(axs[0], "contraste_pct"), (axs[1], "px_cresta_sanas")]:
        y = [res["dedos_val"][t][str(q)][campo] for q in an]
        a.plot(range(4), y, "o-", color=COL_TIPO[t], lw=2, ms=6)
        a.text(3.1, y[-1], NOMBRE_TIPO[t][:4], color=TINTA2, fontsize=8, va="center")
axs[0].set_title("Contraste de los dedos (% del brillo)", loc="left"); axs[1].set_title("Falsa grieta por celda sana (px, media)", loc="left")
for a in axs[:2]: a.set_xticks(range(4), etq); a.set_ylim(0, None); a.set_xlim(-.3, 3.7)
for w, c in zip(an[1:], ["#b39ae8", "#7a5bd0", "#3c2596"]):
    y = [100 * res["retencion_dedos"][str(w)][q] for q in ANG]
    axs[2].plot(ANG, y, "o-", color=c, lw=2, ms=5); axs[2].text(91.5, y[-1], f"1×{w}", fontsize=8, color=TINTA2, va="center")
axs[2].set_title("Contraste de una grieta fina que sobrevive (%)", loc="left"); axs[2].set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[2].set_xlabel("Ángulo de la grieta (0° horizontal · 90° vertical)"); axs[2].set_ylim(0, 105); axs[2].set_xlim(-2, 100)
fig.subplots_adjust(wspace=.25)
guardar(fig, os.path.join(FIG, "b4_dedos.png"))

# b5 · detección v1 frente a v2, y el panel sin rejilla
fig, axs = plt.subplots(2, 5, figsize=(13.4, 5.9))
for c, (i, nom) in enumerate([(923, "mono sana"), (6, "mancha grande"), (2371, "poly sana"), (282, "2 busbars"), (2182, "panel M17, sin rejilla")]):
    In, m, bb = preparar(i)
    v1 = [int(r) for r in busbar_rows(ims[i], m)[1]]; ref = rejilla(part.panel_cod[i])
    for r, filas in [(0, v1), (1, [b["fila"] for b in bb])]:
        a = axs[r, c]; a.imshow(ims[i], cmap="gray"); sin_ejes(a)
        for f in filas:
            ok = any(abs(f - q) <= 9 for q in ref)
            a.axhline(f, color="#12b39b" if ok else "#ff3b30", lw=2.2, ls="-" if ok else "--")
        if r == 0: a.set_title(f"{nom} · #{i}", loc="left", fontsize=9)
axs[0, 0].set_ylabel("v1 (semana 1)", fontsize=10); axs[1, 0].set_ylabel("v2 (semana 2)", fontsize=10)
fig.text(.125, .06, "Verde continuo: busbar correcto. Rojo discontinuo: busbar falso. #2182 (validación): su línea oscura es un defecto, no un busbar.", fontsize=8.5, color=TINTA2)
fig.subplots_adjust(wspace=.05, hspace=.05)
guardar(fig, os.path.join(FIG, "b5_deteccion_v1_v2.png"), dpi=130)

# b6 · qué le hace cada método a una grieta según su ángulo
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for a, z, t in [(axs[0], "fuera", "Grieta lejos de los busbars"), (axs[1], "cruza", "Tramo de la grieta que cruza un busbar")]:
    for k in (METODOS[1:] if z == "fuera" else METODOS):
        d = res[f"retencion_{z}"][k]; a.plot(list(d), [100 * v for v in d.values()], "o--" if k == "A-filas" else "o-", color=COL_M[k], lw=2, ms=5, label=k)
    a.set_title(t, loc="left"); a.set_xlabel("Ángulo de la grieta (0° horizontal, como un busbar · 90° vertical)"); a.set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[0].set_ylabel("Contraste conservado (%)"); axs[0].set_ylim(0, 115); axs[1].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(.02, .42))
guardar(fig, os.path.join(FIG, "b6_grietas_por_angulo.png"))
print(json.dumps({k: v for k, v in res.items() if k != "barrido_train"}, indent=1, default=float, ensure_ascii=False))
