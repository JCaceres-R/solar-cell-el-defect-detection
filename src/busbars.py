"""Semana 2 · Busbars y dedos: detección v2, notch gaussiano (A), cierre morfológico (B) y métricas de imagen.

La línea base (src/preproc.py) NO se toca, para que el Avance 1 siga siendo reproducible.
Todo trabaja sobre In, la celda normalizada (ec. 1 de la propuesta), y sobre su máscara M.

  detectar(In, M)            -> lista de busbars {fila, y0, y1, grosor, contraste, cobertura}
  filas_banda(bb)            -> vector booleano (300,) con las filas de busbar
  notch(In, M, bb)           -> A-global (ec. 2): agujeros gaussianos en (0, ±k·v0), sin tocar DC
  notch_filas(In, M, bb)     -> A-filas: la salida de A solo dentro de las filas de busbar
  cierre(In, bb)             -> B (ec. 3): cierre vertical solo en las filas de busbar
  quitar_dedos(Ib, ancho)    -> cierre horizontal corto (§4.3)
"""
from itertools import combinations

import cv2
import numpy as np

H = W = 300

# Detección v2 (valores fijados mirando solo ENTRENAMIENTO; ver scripts/07_comparar_busbars.py)
H_SOMBRERO = 27        # alto del cierre para el sombrero negro: deja fuera todo lo que sea más grueso (manchas)
CONTRASTE_MIN = 0.06   # hundimiento mediano a lo ancho, como fracción del brillo local
COBERTURA_MIN = 0.60   # fracción del ancho de la celda donde la banda se hunde al menos COL_MIN
COL_MIN = 0.03
GROSOR_MAX = 16        # px
SEP3, SEP2, CENTRO = (90, 106), (132, 158), 12   # geometría de la rejilla (px)
MARGEN = 3             # px que se suman arriba y abajo de la banda detectada

# Parámetros de A y B, elegidos en ENTRENAMIENTO con scripts/07_comparar_busbars.py (barrido)
SIGMA_N = 1.0     # agujeros separados: sigma <= v0/2, y v0 = 2,1 en las celdas de 2 busbars
K_MAX = 24        # armónicos; con más de 20 ya no cambia nada
H_CIERRE = 21     # px; el de menor residual por tramos en entrenamiento (las bandas miden ~8 px)
ANCHO_DEDO = 3    # px; la semana 2 recomienda no aplicarlo (ver entregas/avance_2)


def _lineal_v(h):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (1, h))


def rellenar_fuera(In, M):
    """Fuera de la máscara pone la mediana de la celda, para no meter bordes artificiales."""
    f = In.astype(np.float32).copy()
    f[~M] = np.median(f[M])
    return f


def mapa_valles(In):
    """Sombrero negro vertical relativo: cuánto se hunde cada píxel respecto a lo que tiene arriba y abajo.

    Una banda oscura más fina que H_SOMBRERO da un valor positivo; una mancha más gruesa y la caída de brillo hacia
    el borde de la celda dan cero, porque el cierre no las rellena. Esas eran las dos fallas de busbar_rows() (v1).
    """
    f = cv2.GaussianBlur(In.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=.6)
    c = cv2.morphologyEx(f, cv2.MORPH_CLOSE, _lineal_v(H_SOMBRERO), borderType=cv2.BORDER_REPLICATE)
    return (c - f) / (c + 1e-3)


def candidatos(In, M):
    """Filas que parecen busbar por sí solas: banda fina, hundida y a todo lo ancho."""
    D = mapa_valles(In)
    cols = M[H // 2 - 60:H // 2 + 60].mean(0) > .5          # columnas dentro de la celda
    med = np.median(D[:, cols], axis=1)
    cob = (D[:, cols] > COL_MIN).mean(1)
    filas_ok = M[:, W // 2 - 60:W // 2 + 60].mean(1) > .5
    med = np.where(filas_ok, med, 0)
    bb = []
    for r in range(12, H - 12):
        if med[r] < CONTRASTE_MIN or med[r] < med[r - 12:r + 13].max() or cob[r] < COBERTURA_MIN:
            continue
        if bb and r - bb[-1]["fila"] <= 12:
            continue
        lo = hi = r
        while lo > 0 and med[lo - 1] > .25 * med[r]:
            lo -= 1
        while hi < H - 1 and med[hi + 1] > .25 * med[r]:
            hi += 1
        if hi - lo + 1 > GROSOR_MAX:
            continue
        bb.append({"fila": int(r), "y0": int(lo), "y1": int(hi), "grosor": int(hi - lo + 1),
                   "contraste": float(med[r]), "cobertura": float(cob[r])})
    return bb


def detectar(In, M):
    """Busbars de la celda: el mejor grupo de candidatos que respeta la geometría de la rejilla.

    La rejilla de ELPV tiene 3 busbars (separados ~98 px) o 2 (~143 px), centrados en la celda. Una línea oscura suelta no
    cumple la regla y NO se toma por busbar. Si de una rejilla de 3 falta una banda (tapada por una mancha), se infiere.
    Devuelve [] si la celda no tiene rejilla visible (contacto trasero, panel M17) o está casi apagada.
    """
    c = candidatos(In, M)
    op = []
    for a, b, d in combinations(c, 3):
        p1, p2 = b["fila"] - a["fila"], d["fila"] - b["fila"]
        if all(SEP3[0] <= p <= SEP3[1] for p in (p1, p2)) and abs(b["fila"] - H // 2) <= CENTRO:
            op.append((sum(x["contraste"] for x in (a, b, d)), [a, b, d]))
    for a, b in combinations(c, 2):
        sep, cen = b["fila"] - a["fila"], (a["fila"] + b["fila"]) / 2
        s = a["contraste"] + b["contraste"]
        if SEP2[0] <= sep <= SEP2[1] and abs(cen - H / 2) <= CENTRO:
            op.append((s, [a, b]))
        if SEP3[0] <= sep <= SEP3[1] or 2 * SEP3[0] <= sep <= 2 * SEP3[1]:      # dos bandas de una rejilla de 3
            if 2 * SEP3[0] <= sep:
                f = round(cen)
            else:
                f = 2 * a["fila"] - b["fila"] if abs(a["fila"] - H / 2) <= CENTRO else 2 * b["fila"] - a["fila"]
                if min(abs(a["fila"] - H / 2), abs(b["fila"] - H / 2)) > CENTRO:
                    continue
            if abs(sorted([a["fila"], b["fila"], f])[1] - H / 2) > CENTRO or not 12 <= f < H - 12:
                continue
            g = round((a["grosor"] + b["grosor"]) / 2)
            inf = {"fila": int(f), "y0": int(f - g // 2), "y1": int(f - g // 2 + g - 1), "grosor": int(g),
                   "contraste": 0.0, "cobertura": 0.0, "inferido": True}
            op.append((s, sorted([a, b, inf], key=lambda x: x["fila"])))
    return max(op, key=lambda x: x[0])[1] if op else []


def filas_banda(bb, margen=MARGEN):
    f = np.zeros(H, bool)
    for b in bb:
        f[max(0, b["y0"] - margen):b["y1"] + margen + 1] = True
    return f


def periodo(bb):
    """Separación media entre busbars vecinos (px). None si hay menos de dos."""
    return float(np.diff([b["fila"] for b in bb]).mean()) if len(bb) >= 2 else None


def filtro_notch(P, sigma=None, K=None, n=H):
    """H(u, v) de la ecuación 2, con el espectro centrado (DC en n/2, n/2)."""
    sigma = SIGMA_N if sigma is None else sigma
    K = K_MAX if K is None else K
    v0 = n / P
    v, u = np.mgrid[:n, :n].astype(np.float32) - n // 2
    Hf = np.ones((n, n), np.float32)
    for k in range(1, K + 1):
        if k * v0 > n / 2 - 1:
            break
        for s in (1, -1):
            Hf *= 1 - np.exp(-(u ** 2 + (v - s * k * v0) ** 2) / (2 * sigma ** 2))
    Hf[n // 2, n // 2] = 1
    return Hf


_cache = {}


def notch(In, M, bb, sigma=None, K=None):
    P = periodo(bb)
    f = rellenar_fuera(In, M)
    if P is None:
        return f
    clave = (round(P, 2), SIGMA_N if sigma is None else sigma, K_MAX if K is None else K)
    if clave not in _cache:
        _cache[clave] = np.fft.ifftshift(filtro_notch(P, sigma, K))
    g = np.fft.ifft2(np.fft.fft2(f) * _cache[clave]).real.astype(np.float32)
    g[~M] = f[~M]
    return g


def notch_filas(In, M, bb, sigma=None, K=None):
    g, f = notch(In, M, bb, sigma, K), rellenar_fuera(In, M)
    b = filas_banda(bb)
    out = f.copy(); out[b] = g[b]
    return out


def cierre(In, M, bb, h=None):
    f = rellenar_fuera(In, M)
    c = cv2.morphologyEx(f, cv2.MORPH_CLOSE, _lineal_v(H_CIERRE if h is None else h), borderType=cv2.BORDER_REPLICATE)
    b = filas_banda(bb)
    out = f.copy(); out[b] = c[b]
    return out


def quitar_dedos(Ib, ancho=None):
    a = ANCHO_DEDO if ancho is None else ancho
    if a <= 1:
        return Ib.copy()
    return cv2.morphologyEx(Ib.astype(np.float32), cv2.MORPH_CLOSE, np.ones((1, a), np.uint8), borderType=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- métricas de imagen (tabla 3 de la propuesta)
def residual(I, M, bb, ancho_vecinas=8, tramo=None):
    """Criterio 1: |brillo en las filas del busbar / brillo de sus filas vecinas - 1|, en %.

    tramo=None usa la fila completa; tramo=12 lo mide por tramos de 12 columnas (detecta restos parciales).
    Devuelve (valor absoluto medio, valor con signo medio).
    """
    cols = np.where(M[H // 2 - 60:H // 2 + 60].mean(0) > .5)[0]
    grupos = [cols] if tramo is None else [cols[i:i + tramo] for i in range(0, len(cols) - tramo + 1, tramo)]
    v = []
    for b in bb:
        a0, a1 = max(0, b["y0"] - MARGEN - ancho_vecinas), b["y0"] - MARGEN
        c0, c1 = b["y1"] + MARGEN + 1, min(H, b["y1"] + MARGEN + 1 + ancho_vecinas)
        for g in grupos:
            vec = np.concatenate([I[a0:a1][:, g], I[c0:c1][:, g]]).mean()
            v.append(I[b["fila"] - 1:b["fila"] + 2][:, g].mean() / (vec + 1e-6) - 1)
    v = np.array(v)
    return 100 * float(np.abs(v).mean()), 100 * float(v.mean())


def cambio_fuera(I, In, M, bb):
    """Criterio 2: |salida - entrada| medio dentro de la máscara y fuera de las filas de busbar, en % del brillo sano."""
    m = M.copy(); m[filas_banda(bb)] = False
    return 100 * float(np.abs(I[m] - In[m]).mean())
