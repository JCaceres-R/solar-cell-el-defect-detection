"""Semana 2 · Datos de la página interactiva (entregas/avance_2/web): celdas en PNG y cifras en JSON.

Uso:   python scripts/08_exportar_web.py      (requiere 07_comparar_busbars.py)

Cada PNG guarda la celda cruda en el canal rojo y su máscara en el verde; la página hace la normalización, el notch y el
cierre en vivo. Solo celdas de entrenamiento y validación.
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import numpy as np
import pandas as pd
from PIL import Image

import busbars as B
from datos import cargar_imagenes
from preproc import busbar_rows, cell_masks, image_mask

OUT = os.path.join(RAIZ, "entregas", "avance_2", "web"); os.makedirs(os.path.join(OUT, "celdas"), exist_ok=True)
part = pd.read_csv(os.path.join(RAIZ, "data", "particion.csv"))
ims = cargar_imagenes(); tipos = part.tipo.values
tr = (part.split == "train").values
T = cell_masks(ims[tr], tipos[tr])

# rótulos por inspección visual: [clase, texto, x, y, ancho, alto] en píxeles
CELDAS = {
    923: ("Sana, rejilla punteada", [["nocuenta", "Dedo interrumpido (probable)", 114, 172, 12, 60]]),
    15: ("Grieta diagonal", [["defecto", "Grieta", 194, 54, 100, 96], ["defecto", "Zona apagada", 22, 4, 258, 27],
                             ["nocuenta", "Dedo interrumpido (probable)", 178, 256, 16, 41]]),
    6: ("Zona apagada grande", [["defecto", "Zona apagada", 8, 88, 250, 78], ["defecto", "Grieta", 186, 56, 110, 46]]),
    282: ("Grieta, 2 busbars", [["defecto", "Grieta", 8, 92, 284, 52]]),
    2371: ("Sana, dedos visibles", [["nocuenta", "Dedo interrumpido (probable)", 155, 160, 12, 70],
                                    ["nocuenta", "Dedo interrumpido (probable)", 49, 70, 12, 76]]),
    2182: ("Panel M17, sin busbars", [["defecto", "Línea oscura", 6, 56, 288, 20]]),
}
celdas = []
for i, (nombre, marcas) in CELDAS.items():
    assert part.split[i] != "test"
    m = image_mask(ims[i], T[tipos[i]]); f = ims[i].astype(np.float32); p90 = float(np.percentile(f[m], 90) + 1e-6)
    bb = B.detectar(f / p90, m)
    Image.fromarray(np.dstack([ims[i], (m * 255).astype(np.uint8), ims[i]])).save(os.path.join(OUT, "celdas", f"c{i}.png"))
    In = f / p90
    celdas.append({"id": i, "nombre": nombre, "tipo": tipos[i], "etiqueta": float(part.prob[i]), "parte": part.split[i],
                   "panel": part.panel_cod[i], "p90": p90, "mediana": float(np.median(In[m])),
                   "bb": [{k: b[k] for k in ("fila", "y0", "y1")} | {"inferido": bool(b.get("inferido", False))} for b in bb],
                   "v1": [int(r) for r in busbar_rows(ims[i], m)[1]], "marcas": marcas,
                   # control: la página debe reproducir estos valores
                   "control": {k: [round(x, 4) for x in (B.residual(I, m, bb)[0], B.cambio_fuera(I, B.rellenar_fuera(In, m), m, bb))]
                               for k, I in (("A-global", B.notch(In, m, bb)), ("A-filas", B.notch_filas(In, m, bb)),
                                            ("B", B.cierre(In, m, bb)))} if bb else {}})
R = json.load(open(os.path.join(RAIZ, "reports", "metricas", "semana2_busbars.json"), encoding="utf-8"))
datos = {"celdas": celdas, "parametros": R["parametros"],
         "criterios": {t: R["criterios_val"][t] for t in ("todas", "mono", "poly")}, "tiempo": R["tiempo_ms"],
         "n": {"todas": R["criterios_val"]["n_celdas"], **R["criterios_val"]["n_por_tipo"]},
         "deteccion": {k: R["deteccion"][k] for k in ("train_todas", "val_todas", "val_mono", "val_poly")},
         "ret": {z: R[f"retencion_{z}"] for z in ("fuera", "cruza", "dedos")},
         "dedos": R["dedos_val"]["todas"], "dedos_tipo": {t: R["dedos_val"][t] for t in ("mono", "poly")}}
json.dump(datos, open(os.path.join(OUT, "datos.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"), default=float)
print([(c["id"], len(c["bb"]), c["control"]) for c in celdas]); print(os.path.getsize(os.path.join(OUT, "datos.json")) // 1024, "KB")

# página autocontenida: plantilla + datos + imágenes incrustadas (se abre con doble clic, sin servidor)
import base64
for c in datos["celdas"]:
    c["png"] = "data:image/png;base64," + base64.b64encode(open(os.path.join(OUT, "celdas", f"c{c['id']}.png"), "rb").read()).decode()
html = open(os.path.join(OUT, "plantilla.html"), encoding="utf-8").read().replace("/*DATOS*/", json.dumps(datos, ensure_ascii=False, separators=(",", ":"), default=float), 1)
open(os.path.join(OUT, "laboratorio_busbars.html"), "w", encoding="utf-8", newline="\n").write(html)
