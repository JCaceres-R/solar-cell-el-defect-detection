"""Estilo común de las figuras del proyecto (para que todas se vean como un mismo sistema).

Colores con un solo significado en todo el proyecto:
  - tipo de celda:  mono = azul, poly = naranja
  - parte del dataset: train = aguamarina, val = amarillo, test = magenta
  - etiqueta 0/33/67/100 %: escala de un solo tono (violeta, de claro a oscuro), porque es ordinal
"""
import matplotlib.pyplot as plt

TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#d6d4ce"
COL_TIPO = {"mono": "#2a78d6", "poly": "#eb6834"}
NOMBRE_TIPO = {"mono": "monocristalina", "poly": "policristalina"}
COL_PARTE = {"train": "#1baf7a", "val": "#eda100", "test": "#e87ba4"}
NOMBRE_PARTE = {"train": "entrenamiento", "val": "validación", "test": "prueba"}
COL_CLASE4 = ["#cfc9f3", "#9a8ee6", "#6250d6", "#33278f"]
NOMBRE_CLASE4 = ["0 %", "33 %", "67 %", "100 %"]


def aplicar_estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlecolor": TINTA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
        "axes.edgecolor": GRIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": 0.6,
    })


def guardar(fig, ruta, dpi=160):
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
