"""Genera los metadatos por celda y la partición train / val / test por panel.

Uso (desde la raíz del repo):   python scripts/00_crear_particion.py

Salidas (todas versionadas en git; son pequeñas):
  data/metadatos.csv          una fila por celda: etiquetas, tipo, panel, grupo, brillo
  data/particion.csv          LA partición congelada del proyecto (idx, archivo, ..., split)
  data/particion_resumen.csv  conteos por parte, tipo y clase
  data/particion_info.json    semilla, puntaje y verificación de fuga
  reports/figures/p*.png      figuras de la partición

Es determinista: con la misma semilla produce exactamente el mismo archivo.
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

from datos import PANELES_GEMELOS, cargar_imagenes, metadatos
from estilo import COL_PARTE, NOMBRE_PARTE, aplicar_estilo, guardar
from particion import (PARTES, crear_particion, fuga, matriz_similitud,
                       pares_casi_identicos, resumen)

DATA = os.path.join(RAIZ, "data")
FIG = os.path.join(RAIZ, "reports", "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)
aplicar_estilo()

# 1. metadatos ----------------------------------------------------------------------------------
meta = metadatos()
meta.to_csv(os.path.join(DATA, "metadatos.csv"), index=False)

# 2. partición ----------------------------------------------------------------------------------
split, info = crear_particion(meta)
meta["split"] = split.values
cols = ["idx", "archivo", "tipo", "panel_cod", "grupo", "prob", "clase4", "defectuosa", "clase2", "split"]
meta[cols].to_csv(os.path.join(DATA, "particion.csv"), index=False)
R = resumen(meta, split)
R.to_csv(os.path.join(DATA, "particion_resumen.csv"), index=False)

# 3. verificación de fuga: pares casi idénticos que quedan en partes distintas -------------------
ims = cargar_imagenes()
C = matriz_similitud(ims, meta.tipo.values)
oscura = meta.oscura.values.astype(bool)
# referencia: lo que pasaría con una partición AL AZAR por celda (estratificada, mismas proporciones)
azar = np.empty(len(meta), dtype=object)
rng = np.random.default_rng(0)
for _, g in meta.groupby(["tipo", "clase4"]):           # 70 / 15 / 15 dentro de cada tipo y clase
    i = rng.permutation(g.idx.values); a, b = int(.70 * len(i)), int(.85 * len(i))
    azar[i[:a]], azar[i[a:b]], azar[i[b:]] = "train", "val", "test"

info["fuga"] = {}
for u in [0.90, 0.95]:
    pr = pares_casi_identicos(C, oscura, u)
    info["fuga"][f"r>{u}"] = {"por_panel": fuga(pr, meta.split.values), "al_azar_por_celda": fuga(pr, azar),
                             "pares_mismo_panel": int((meta.panel.values[pr[:, 0]] == meta.panel.values[pr[:, 1]]).sum())}
info["paneles"] = {p: sorted(meta[meta.split == p].panel_cod.unique().tolist()) for p in PARTES}
json.dump(info, open(os.path.join(DATA, "particion_info.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# 4. figuras ------------------------------------------------------------------------------------
# p1: composición de cada parte
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.3), gridspec_kw={"width_ratios": [1, 1.1], "wspace": .35})
ax = axs[0]
for i, p in enumerate(PARTES):
    r = R[R.parte == p].iloc[0]
    ax.barh(i, r.FUNCIONAL, color="#9fb7d9", height=.55)
    ax.barh(i, r.DEFECTUOSA, left=r.FUNCIONAL + 6, color="#2a78d6", height=.55)
    ax.text(r.FUNCIONAL + r.DEFECTUOSA + 30, i, f"{r.celdas} celdas · {r.paneles} paneles · {r['% defectuosa']} % def.",
            va="center", fontsize=8, color="#52514e")
ax.set_yticks(range(3), [NOMBRE_PARTE[p] for p in PARTES]); ax.invert_yaxis()
ax.set_xlim(0, 3300); ax.set_xticks([0, 500, 1000, 1500, 2000]); ax.set_xlabel("Nº de celdas")
ax.set_title("Celdas por parte (claro = funcional, oscuro = defectuosa)", loc="left", fontsize=9)
ax = axs[1]
x = np.arange(4); w = .26
for i, p in enumerate(PARTES):
    r = R[R.parte == p].iloc[0]
    v = np.array([r["0 %"], r["33 %"], r["67 %"], r["100 %"]]) / r.celdas * 100
    ax.bar(x + (i - 1) * w, v, w - .03, color=COL_PARTE[p], label=NOMBRE_PARTE[p])
tot = R[R.parte == "total"].iloc[0]
vt = np.array([tot["0 %"], tot["33 %"], tot["67 %"], tot["100 %"]]) / tot.celdas * 100
ax.scatter(x, vt, marker="_", s=500, color="#0b0b0b", zorder=3, label="dataset completo")
ax.set_xticks(x, ["0 %", "33 %", "67 %", "100 %"]); ax.set_xlabel("Probabilidad de defecto (etiqueta del experto)")
ax.set_ylabel("% de las celdas de la parte"); ax.legend(frameon=False, ncol=1, fontsize=8, loc="upper right")
ax.set_title("Mezcla de etiquetas de cada parte vs. el dataset completo", loc="left", fontsize=9)
guardar(fig, os.path.join(FIG, "p1_particion_composicion.png"))

# p2: mapa de los 44 paneles: % de celdas defectuosas y parte asignada
P = meta.groupby("panel").agg(cod=("panel_cod", "first"), tipo=("tipo", "first"), split=("split", "first"),
                              n=("idx", "size"), defe=("defectuosa", "mean"), exacto=("panel_exacto", "all")).reset_index()
fig, ax = plt.subplots(figsize=(11, 3.0))
gemelos = {c for par in PANELES_GEMELOS for c in par}
for _, r in P.iterrows():
    h = "////" if not r.exacto else ("...." if r.cod in gemelos else None)
    ax.bar(r.panel, 100 * r.defe, .8, color=COL_PARTE[r.split], hatch=h, edgecolor="#3d3c39" if h else "white", linewidth=0)
ax.set_xticks(P.panel, P.cod, rotation=90, fontsize=6.5)
ax.set_ylabel("% de celdas defectuosas")
ax.set_xlim(-.7, 43.7)
for p in PARTES:
    ax.bar(np.nan, np.nan, color=COL_PARTE[p], label=NOMBRE_PARTE[p])
ax.bar(np.nan, np.nan, color="white", hatch="////", edgecolor="#3d3c39", label="frontera no exacta (un solo grupo)")
ax.bar(np.nan, np.nan, color="white", hatch="....", edgecolor="#3d3c39", label="panel gemelo (va con su par)")
ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper left")
ax.text(0, -0.42, "Gemelos (mismo panel fotografiado dos veces): " + " · ".join(f"{a} = {b}" for a, b in PANELES_GEMELOS)
        + "   ·   P22 se une al grupo ambiguo por ser gemelo de P14.", transform=ax.transAxes, fontsize=8, color="#52514e")
ax.set_ylim(0, 115)
ax.set_title("Los 44 paneles reconstruidos (M = mono, P = poly), en el orden del dataset, y la parte a la que fue cada uno", loc="left")
guardar(fig, os.path.join(FIG, "p2_paneles_particion.png"))

print(R.to_string(index=False))
print(json.dumps(info, indent=1, ensure_ascii=False))
