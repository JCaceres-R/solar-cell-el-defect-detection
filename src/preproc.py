"""Pipeline de preprocesamiento para celdas EL (ELPV).

Origen: código de arranque del 27-oct-2026 (investigacion_proyecto_final/P6_codigo/preproc.py), sin cambios
de lógica. Es la LÍNEA BASE de la fase 2-3: aquí es donde el equipo debe mejorar (notch en Fourier para
busbars, corrección de iluminación, grietas en poly). Ver docs/02_eda.md, sección "Decisiones".


Etapas:
  1. Máscara geométrica de la celda (bordes y esquinas achaflanadas) a partir de la imagen media por tipo.
  2. Normalización fotométrica: dividir por el nivel de brillo robusto de la celda (percentil 90) -> 1.0 = celda sana.
  3. Detección adaptativa de busbars (valles del perfil por fila) y borrado por cierre morfológico vertical.
  4. Supresión de dedos (líneas verticales finas, muy tenues) con cierre horizontal corto.
  5. Mapa de zona inactiva: déficit de brillo respecto a 1.0 (fondo suavizado).
  6. Mapa de grietas: filtro de crestas oscuras (Sato / Hessiano) en escalas finas, sin busbars.
  7. Características escalares para describir la celda.
"""
import numpy as np, cv2
from skimage.filters import sato
from skimage import morphology, measure

H = W = 300

def cell_masks(ims, types):
    """Máscara de celda por tipo (mono tiene esquinas achaflanadas)."""
    out = {}
    for t in ["mono", "poly"]:
        m = np.median(ims[types == t].astype(np.float32), axis=0)
        m = cv2.GaussianBlur(m, (0, 0), 2)
        thr = 0.6 * np.percentile(m, 90)
        mk = m > thr
        # quitar busbars del umbral (son valles horizontales): cierre vertical
        mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8)).astype(bool)
        mk = cv2.erode(mk.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        mk[:14, :] = False; mk[-14:, :] = False; mk[:, :14] = False; mk[:, -14:] = False
        out[t] = mk
    return out

def busbar_rows(img, mask):
    """Filas de busbar: valles del perfil medio por fila respecto a su tendencia."""
    f = img.astype(np.float32)
    prof = np.array([f[r, mask[r]].mean() if mask[r].sum() > 50 else np.nan for r in range(H)])
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    prof = np.convolve(prof, np.ones(3) / 3, "same")
    base = cv2.medianBlur(prof.astype(np.float32).reshape(-1, 1), 41).ravel() if False else \
        np.array([np.median(prof[max(0, r - 20):r + 21]) for r in range(H)])
    d = (base - prof) / (base + 1e-6)
    rows = np.zeros(H, bool)
    peaks = [r for r in range(15, H - 15) if d[r] > 0.04 and d[r] == d[r - 8:r + 9].max()]
    for r in peaks:
        lo, hi = r, r
        while lo > 0 and d[lo - 1] > 0.2 * d[r]: lo -= 1
        while hi < H - 1 and d[hi + 1] > 0.2 * d[r]: hi += 1
        rows[max(0, lo - 3):hi + 4] = True
    return rows, peaks

def image_mask(img, template):
    """Máscara propia de la celda: región brillante -> envolvente convexa -> erosión; AND con la plantilla del tipo."""
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    thr = 0.5 * np.percentile(f, 90)
    b = (f > thr).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n > 1:
        b = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(b))
    hm = np.zeros_like(b); cv2.fillConvexPoly(hm, hull, 1)
    hm = cv2.erode(hm, np.ones((17, 17), np.uint8)).astype(bool)
    # esquinas achaflanadas (mono): recortar con la plantilla erosionada solo en las esquinas
    return hm & template

def preprocess(img, mask):
    f = img.astype(np.float32)
    # 2. normalización fotométrica
    ref = np.percentile(f[mask], 90) + 1e-6
    n = f / ref
    # 3. busbars: detectar filas (perfil) y cerrar solo en esa banda con un elemento vertical
    rows, peaks = busbar_rows(img, mask)
    band = cv2.dilate(rows.astype(np.uint8)[:, None].repeat(W, 1), np.ones((9, 1), np.uint8)).astype(bool)
    closed = cv2.morphologyEx(n, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 27)))
    nb = n.copy(); nb[band] = closed[band]
    # 4. dedos: líneas verticales finas -> cierre horizontal corto
    nf = cv2.morphologyEx(nb, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    nf = np.clip(nf, 0, 1.3)
    # fuera de la celda: rellenar con el valor típico para no crear bordes artificiales
    fill = np.median(nf[mask]); nf[~mask] = fill
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    # 5. mapa de zona inactiva (déficit de brillo suavizado)
    sm = cv2.GaussianBlur(nf, (0, 0), 4)
    dark = np.clip(1.0 - sm / fill, 0, 1); dark[~inner] = 0
    # 6. mapa de grietas (crestas oscuras finas)
    rid = sato(nf, sigmas=[1, 2], black_ridges=True); rid[~inner] = 0
    return {"norm": n, "nobus": nf, "dark": dark, "ridge": rid, "rows": rows, "peaks": peaks, "ref": ref, "inner": inner}

def features(pp, mask):
    d, r = pp["dark"], pp["ridge"]
    m = pp["inner"]
    feats = {}
    for th in [0.15, 0.3, 0.5]:
        feats[f"area_oscura_{int(th*100)}"] = float((d[m] > th).mean())
    feats["deficit_medio"] = float(d[m].mean())
    feats["deficit_p99"] = float(np.percentile(d[m], 99))
    # grietas: umbral relativo + componentes alargadas
    cr = crack_mask(pp, m)
    lab = measure.label(cr)
    L = 0; maxL = 0
    for p in measure.regionprops(lab):
        horiz_larga = abs(np.sin(p.orientation)) > 0.97 and p.axis_major_length > 120  # resto de busbar
        if p.axis_major_length > 18 and p.eccentricity > 0.95 and not horiz_larga:
            L += p.axis_major_length; maxL = max(maxL, p.axis_major_length)
    feats["long_grietas"] = float(L); feats["grieta_max"] = float(maxL)
    feats["cresta_media"] = float(r[m].mean())
    feats["std_norm"] = float(pp["nobus"][m].std())
    feats["brillo_ref"] = float(pp["ref"])
    feats["n_busbars"] = len(pp["peaks"])
    return feats, cr

def crack_mask(pp, mask, k=6.0):
    r = pp["ridge"]; v = r[mask]
    med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6
    cr = ((r - med) / (1.4826 * mad) > k) & mask
    return morphology.remove_small_objects(cr, max_size=25)
