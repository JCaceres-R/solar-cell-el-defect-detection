# Detección de celdas solares defectuosas en imágenes de electroluminiscencia

Proyecto final · Procesamiento Digital de Imágenes · 2026-2

**Equipo:** Johan Sebastián Cáceres Rodríguez · Hosman · Joel Eduardo Reyes Barrios

## Qué hace

Recibe la imagen de **electroluminiscencia (EL)** de una celda solar, una especie de "radiografía" del panel donde las grietas y las zonas desconectadas se ven oscuras. Con eso:

1. limpia y normaliza la imagen: corrige el brillo, quita los busbars y los dedos;
2. realza los defectos (grietas y zonas apagadas) y genera un **mapa de defectos**;
3. una CNN pequeña, el único bloque de red del sistema, clasifica la celda como **FUNCIONAL o DEFECTUOSA**.

El procesamiento de imagen hace el trabajo pesado; la red es la última etapa. Datos: el dataset público **ELPV** (2.624 celdas de 44 paneles).

## Estado

| Fase | Estado |
|---|---|
| F0 · Repositorio, entorno, dataset | ✅ |
| F1 · EDA y partición train/val/test por panel | ✅ congelada en `data/particion.csv` |
| F2 · Preprocesamiento (máscara, normalización, busbars, dedos) | 🟡 línea base portada (`src/preproc.py`); faltan mejoras |
| F3 · Mapa de defectos + características | 🟡 línea base portada; falta volver a medirla sobre la partición nueva |
| F4 · CNN | ⬜ plantilla lista (`scripts/04_entrenar_cnn.py`), sin entrenar |
| F5 · Evaluación en test, errores, informe, demo | ⬜ |

## Estructura

```
├── data/
│   ├── metadatos.csv          una fila por celda: etiquetas, tipo, panel reconstruido, grupo, brillo
│   ├── particion.csv          ★ partición congelada (train / val / test)
│   ├── particion_resumen.csv  conteos por parte
│   ├── particion_info.json    semilla y verificación de fuga
│   └── interim/               (no versionado) características y tensores que generan los scripts
├── docs/
│   ├── 01_dataset.md          ficha del dataset: origen, licencia, etiquetas, paneles, calidad, diccionario
│   ├── 02_eda.md              hallazgos del EDA → decisiones
│   └── 03_particion.md        método de partición, resultado y verificación de fuga
├── notebooks/
│   └── 01_EDA.ipynb           análisis exploratorio completo (ejecutado, con salidas)
├── reports/
│   ├── figures/               e* = EDA · p* = partición · f* = pipeline
│   └── metricas/              (se crea al evaluar) JSON con métricas
├── scripts/                   se corren en orden desde la raíz del repo
│   ├── 00_crear_particion.py      metadatos + partición + figuras p*
│   ├── 01_preprocesar_todo.py     pipeline sobre todas las celdas → data/interim/
│   ├── 02_evaluar_caracteristicas.py  AUC por característica + logística (train → val)
│   ├── 03_figuras_pipeline.py     figura paso a paso + errores en val
│   └── 04_entrenar_cnn.py         CNN (en máquina con GPU / Colab)
├── src/
│   ├── datos.py               carga, metadatos, reconstrucción de paneles, gemelos
│   ├── particion.py           partición por grupos, similitud entre celdas/paneles, fuga
│   ├── preproc.py             pipeline de preprocesamiento (línea base)
│   └── estilo.py              colores y estilo común de figuras
├── requirements.txt           dependencias (versiones exactas probadas)
└── requirements-cnn.txt       + torch/torchvision para la fase 4
```

## Cómo reproducir

Requiere Python 3.13 (probado con 3.13.14).

```bash
git clone <url-del-repo> && cd proyecto-final-pdi-celdas-el
python -m venv .venv
# Windows: .venv\Scripts\activate     ·     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt        # instala también el dataset (~93 MB)

python scripts/00_crear_particion.py   # ~4 min · debe reproducir data/particion.csv sin cambios (git diff vacío)
python -m nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb   # ~3 min
python scripts/01_preprocesar_todo.py  # ~3-5 min · LIMITE=40 para una prueba rápida
python scripts/02_evaluar_caracteristicas.py
python scripts/03_figuras_pipeline.py
```

Fase 4, en la máquina con GPU o en Colab:

```bash
pip install -r requirements-cnn.txt    # instalar torch según la CUDA de esa máquina
python scripts/01_preprocesar_todo.py  # data/interim/ no se versiona: hay que generarlo allí
python scripts/04_entrenar_cnn.py pre 0
```

## Datos y partición, en una tabla

| | Paneles | Celdas | FUNCIONAL | DEFECTUOSA | % def. |
|---|---|---|---|---|---|
| train | 30 | 1.790 | 1.262 | 528 | 29,5 |
| val | 7 | 432 | 277 | 155 | 35,9 |
| test | 7 | 402 | 264 | 138 | 34,3 |
| **total** | **44** | **2.624** | **1.803** | **821** | **31,3** |

- La partición es **por panel**, con los paneles gemelos juntos. Ningún par de celdas casi idénticas (r > 0,95) queda repartido entre partes; con una partición al azar quedarían 498. Detalle en [`docs/03_particion.md`](docs/03_particion.md).
- **test no se toca** hasta la evaluación final.

## Flujo de trabajo con git

- `main` siempre debe correr. Cada quien trabaja en una rama: `git checkout -b preproc-notch`, y luego hace un pull request (o merge) a `main`.
- Commits pequeños con mensajes que digan qué cambió y por qué ("busbars: notch en Fourier en vez de cierre morfológico").
- **No se versionan:** imágenes del dataset, `data/interim/`, modelos (`*.pt`, `*.onnx`), `.venv/`. Ver `.gitignore`.
- **Sí se versionan:** código, `data/*.csv` (son pequeños y definen la partición), figuras de `reports/` y docs.
- Si un cambio altera `data/particion.csv`, **no se sube**: la partición es fija.

## Licencias y citas

- Imágenes ELPV: **CC BY-NC-SA 4.0** (uso académico, citando). El repo no las redistribuye; se descargan con `pip`.
- Citar: Buerhop-Lutz et al., EU PVSEC 2018, y Deitsch et al., *Solar Energy* 185 (2019) 455–468. Referencias completas en [`docs/01_dataset.md`](docs/01_dataset.md).
