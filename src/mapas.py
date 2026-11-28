"""Semana 3 · Los dos mapas de la propuesta (§4.4) y la entrada de la CNN (§4.5).

Todo trabaja sobre Ic, la celda limpia: In (ec. 1) con los busbars quitados por A o B (src/busbars.py).

  limpiar(In, M, bb, metodo, dedos)  -> Ic según el método: "In" (sin quitar nada), "A-global", "A-filas" o "B"
  deficit(Ic, M)                      -> D (ec. 5): fracción de brillo que falta respecto a la mediana de la celda
  sato_crestas(Ic, M)                 -> respuesta de Sato (ec. 4) a líneas oscuras finas
  gabor_crestas(Ic, M)                -> plan B para poly: banco de Gabor alargados, máximo sobre orientaciones
  grietas(R, n, ...)                  -> máscara de grietas: umbral robusto + filtro por longitud y rectitud
  entrada_cnn(Ic, D, R)               -> los 3 canales de 128 × 128 que ve la red
"""
import cv2
import numpy as np
from skimage import measure, morphology
from skimage.filters import sato

import busbars as B

H = W = 300

# Parámetros fijados en ENTRENAMIENTO con scripts/10_mapas.py (barrido); ver reports/metricas/semana3_mapas.json
SIGMA_D = 2.0           # px; suavizado del mapa de déficit (barrido 2, 4, 8)
# Zona apagada: D por encima del umbral. Por tipo, el de mayor AUC en entrenamiento entre los que dejan <= 1 % de área en las
# celdas sanas. En poly hace falta uno más alto: con 0,1-0,2 el mapa pinta el grano oscuro de las celdas sanas.
UMBRAL_D = {"mono": 0.15, "poly": 0.40}
SIGMAS_SATO = (1, 2)    # px; escalas finas, como la línea base (σ 1-3 y el banco de Gabor separan peor)
K_MAD = 4.0             # umbral robusto sobre la respuesta: mediana + K·MAD (barrido 4, 6, 8)
LARGO_MIN = 30          # px; las componentes más cortas no cuentan como grieta (barrido 18, 30)
RECTITUD_MIN = 0.0      # largo del eje mayor / largo del esqueleto (1 = recta). 0 = sin filtro: con 0,8 separa peor
CANAL_MAX = {"Ic": 1.2, "D": 1.0, "R": 0.15}   # escala fija de cada canal antes de pasar a uint8


def interior(M, e=9):
    return cv2.erode(M.astype(np.uint8), np.ones((e, e), np.uint8)).astype(bool)


def limpiar(In, M, bb, metodo="B", dedos=1):
    if metodo == "In":
        I = B.rellenar_fuera(In, M)
    elif metodo == "A-global":
        I = B.notch(In, M, bb)
    elif metodo == "A-filas":
        I = B.notch_filas(In, M, bb)
    elif metodo == "B":
        I = B.cierre(In, M, bb)
    else:
        raise ValueError(metodo)
    I = B.quitar_dedos(I, dedos)
    I = np.clip(I, 0, 1.3).astype(np.float32)
    I[~M] = np.median(I[M])
    return I


def deficit(Ic, M, sigma=None):
    """Ec. 5: D = max(0, 1 - (G_s * Ic) / mediana(Ic en M)). Vale 0 en lo sano y 0,6 en una zona al 40 %."""
    s = SIGMA_D if sigma is None else sigma
    g = cv2.GaussianBlur(Ic, (0, 0), s)
    D = np.clip(1 - g / (np.median(Ic[M]) + 1e-6), 0, 1)
    D[~interior(M)] = 0
    return D.astype(np.float32)


def area_apagada(D, M, tipo, umbral=None):
    n = interior(M)
    return float((D[n] > (UMBRAL_D[tipo] if umbral is None else umbral)).mean())


def sato_crestas(Ic, M, sigmas=None):
    r = sato(Ic, sigmas=list(SIGMAS_SATO if sigmas is None else sigmas), black_ridges=True).astype(np.float32)
    r[~interior(M)] = 0
    return r


_gabor = {}


def gabor_crestas(Ic, M, lambd=8.0, sigma=2.0, gamma=0.25, n_ang=12):
    """Banco de Gabor par (simétrico): cada núcleo es una línea oscura corta con su orientación.

    gamma < 1 alarga el núcleo a lo largo de la línea: premia los trazos rectos y castiga los que cambian de dirección,
    que es la diferencia entre una grieta y un borde de grano (§4.4). Se toma el máximo sobre las orientaciones.
    """
    clave = (lambd, sigma, gamma, n_ang)
    if clave not in _gabor:
        k = int(2 * np.ceil(3 * sigma / gamma) + 1)
        ks = []
        for t in np.arange(n_ang) * np.pi / n_ang:
            g = cv2.getGaborKernel((k, k), sigma, t, lambd, gamma, 0, ktype=cv2.CV_32F)
            ks.append(-(g - g.mean()) / np.abs(g).sum())       # signo: línea OSCURA da respuesta positiva
        _gabor[clave] = ks
    f = Ic - cv2.GaussianBlur(Ic, (0, 0), 6)                    # quita la iluminación de baja frecuencia
    r = np.max([cv2.filter2D(f, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT) for k in _gabor[clave]], axis=0)
    r = np.clip(r, 0, None)
    r[~interior(M)] = 0
    return r


def _rectitud(p):
    sk = morphology.skeletonize(p.image)
    return p.axis_major_length / max(sk.sum(), 1)


def grietas(R, n, k=None, largo_min=None, rectitud_min=None):
    """Máscara de grietas: (R - mediana) / MAD > k, y solo las componentes largas y rectas.

    Devuelve (máscara, largo total en px, largo de la mayor).
    """
    k = K_MAD if k is None else k
    lm = LARGO_MIN if largo_min is None else largo_min
    rm = RECTITUD_MIN if rectitud_min is None else rectitud_min
    v = R[n]; med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
    c = ((R - med) / (1.4826 * mad) > k) & n
    c = morphology.remove_small_objects(c, max_size=25)
    lab = measure.label(c)
    out = np.zeros_like(c); L = 0.0; Lmax = 0.0
    for p in measure.regionprops(lab):
        if p.axis_major_length < lm:
            continue
        if rm > 0 and _rectitud(p) < rm:
            continue
        out[lab == p.label] = True
        L += p.axis_major_length; Lmax = max(Lmax, p.axis_major_length)
    return out, L, Lmax


def entrada_cnn(Ic, D, R, lado=128):
    """3 canales uint8 de lado × lado: celda limpia, déficit y crestas, cada uno con su escala fija."""
    x = [np.clip(z / CANAL_MAX[c], 0, 1) for z, c in ((Ic, "Ic"), (D, "D"), (R, "R"))]
    x = np.stack([cv2.resize(z.astype(np.float32), (lado, lado), interpolation=cv2.INTER_AREA) for z in x])
    return (x * 255).round().astype(np.uint8)
