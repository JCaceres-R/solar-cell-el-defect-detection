# 01 · Ficha del dataset (ELPV)

> Documento de referencia sobre **qué datos usa el proyecto, de dónde salen, qué significan y qué limitaciones tienen**.
> Los números de este documento salen de `notebooks/01_EDA.ipynb` y `scripts/00_crear_particion.py`; si se regeneran, deben coincidir.

## 1. Identificación

| Campo | Valor |
|---|---|
| Nombre | ELPV — *elpv-dataset* (Electroluminescence Photovoltaic) |
| Versión usada | paquete `elpv-dataset==1.0.0.post1` (PyPI) |
| Autores / institución | ZAE Bayern (Forschungszentrum Erlangen-Nürnberg), FAU Erlangen-Nürnberg y colaboradores |
| Repositorio | https://github.com/zae-bayern/elpv-dataset |
| Licencia de las imágenes | **CC BY-NC-SA 4.0**: uso académico permitido citando la fuente; sin uso comercial; obras derivadas con la misma licencia |
| Licencia del cargador (`utils.py`) | Apache-2.0 |
| Acceso | `pip install elpv-dataset` — sin registro, sin formularios, ~93 MB |

**Cómo citar** (obligatorio por la licencia):

- Buerhop-Lutz, C.; Deitsch, S.; Maier, A.; Gallwitz, F.; Berger, S.; Doll, B.; Hauch, J.; Camus, C.; Brabec, C. J. *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery*. 35th EU PVSEC, 2018, pp. 1287–1289. DOI: 10.4229/35thEUPVSEC20182018-5CV.3.15
- Deitsch, S.; Christlein, V.; Berger, S.; Buerhop-Lutz, C.; Maier, A.; Gallwitz, F.; Riess, C. *Automatic classification of defective photovoltaic module cells in electroluminescence images*. Solar Energy 185 (2019) 455–468. DOI: 10.1016/j.solener.2019.02.067
- (Recorte de celdas) Deitsch, S. et al. *Segmentation of photovoltaic module cells in uncalibrated electroluminescence images*. Machine Vision and Applications 32, 84 (2021). DOI: 10.1007/s00138-021-01191-9

**Este repositorio no redistribuye las imágenes**: se descargan con `pip`. En git solo hay metadatos derivados (CSV) y figuras.

## 2. Cómo se obtuvieron las imágenes

- **Electroluminiscencia (EL):** se inyecta corriente al panel en un cuarto oscuro y la celda emite una luz infrarroja débil que capta una cámara sensible al infrarrojo cercano. Las zonas sanas brillan; las grietas, las zonas eléctricamente desconectadas y otros defectos se ven oscuros. Es como una radiografía del panel.
- Los autores fotografiaron **44 paneles** (18 monocristalinos y 26 policristalinos), corrigieron la distorsión del lente, recortaron cada celda, la enderezaron (perspectiva) y la llevaron a 300×300 px.
- **Consecuencia para el proyecto:** las imágenes ya vienen recortadas y alineadas. La parte geométrica (encontrar las celdas en el panel completo) no forma parte del proyecto.
- **No podemos tomar imágenes propias:** hace falta una fuente de corriente, una cámara IR y un cuarto oscuro. Las "imágenes nuevas" de la demo son las celdas de **prueba**, de paneles que el modelo nunca vio (ver `03_particion.md`).

## 3. Contenido y formato

| Elemento | Detalle |
|---|---|
| Imágenes | 2.624 PNG en `elpv_dataset/data/images/cell0001.png … cell2624.png` |
| Formato | 300 × 300 px, escala de grises de 8 bits (`uint8`, rango 0–255) |
| Peso | 90,3 MB en disco |
| Etiquetas | `labels.csv`: `ruta  probabilidad  tipo` (separado por espacios), una fila por imagen |
| Faltantes | 0 |
| Imágenes exactamente repetidas | 0 |

Cargar en el proyecto:

```python
import sys; sys.path.insert(0, "src")
from datos import cargar_imagenes, metadatos
ims = cargar_imagenes()     # (2624, 300, 300) uint8
meta = metadatos()          # DataFrame con una fila por celda (ver §9)
```

## 4. Etiquetas

Un experto miró cada celda y dio una **probabilidad de defecto**, que combina su decisión con su seguridad:

| Etiqueta original | Qué dijo el experto | Clase en el proyecto (`clase2`) | `clase4` |
|---|---|---|---|
| 0 | Funcional, seguro | FUNCIONAL | 0 |
| 0,33 | Funcional, con duda | FUNCIONAL | 1 |
| 0,67 | Defectuosa, con duda | DEFECTUOSA | 2 |
| 1 | Defectuosa, seguro | DEFECTUOSA | 3 |

- Pasar de 4 a 2 clases conserva la decisión del experto y descarta la duda. Es el mismo criterio de la literatura (umbral 0,5).
- **Defectos incluidos:** microgrietas, zonas eléctricamente aisladas o degradadas, celdas en cortocircuito, interconexiones abiertas y fallas de soldadura, es decir, defectos que reducen la potencia. **No cuentan** los dedos interrumpidos.
- **Limitaciones:** hubo un solo experto, no hay medición de potencia que confirme el daño y no se marca dónde está el defecto (solo la etiqueta global de la celda).

## 5. Paneles reconstruidos (columna que el dataset no trae)

El dataset no dice de qué panel viene cada celda. Hace falta saberlo para partir los datos sin fuga, así que se reconstruye en `src/datos.py` (tabla `TRAMOS`):

1. Las celdas vienen en orden de panel. El tipo (mono/poly) cambia en 10 tramos, y casi todos miden un múltiplo exacto de 60, porque los paneles son de 60 celdas.
2. Al contar salen exactamente **18 paneles mono y 26 poly**, lo mismo que publicaron los autores. Para que cuadre, los tramos de 72 y 48 celdas tienen que ser un panel cada uno, y debe haber un panel incompleto en el tramo mono de 714 celdas y otro en el tramo poly de 530.
3. Cada panel se fotografió con su propia exposición, así que **el brillo medio salta en las fronteras** (figuras `e3` y `e4`). Con eso se ubica el panel mono incompleto: **M10, celdas 1200–1253 (54 celdas)**.
4. Entre las celdas 1674 y 1963 hay 5 paneles poly, P13 a P17, y uno de ellos tiene 50 celdas, pero no hay saltos claros para ubicarlo. Por eso se marcan como `panel_exacto = False` y se tratan como **un solo grupo** al partir.

| Tipo | Paneles | Celdas | Tamaños |
|---|---|---|---|
| mono | 18 | 1.074 | 15 de 60, 1 de 54 (M10), 1 de 72 (M17), 1 de 48 (M18) |
| poly | 26 | 1.550 | 25 de 60, 1 de 50 (dentro de P13–P17) |

Códigos: `M01…M18` (mono) y `P01…P26` (poly), numerados en el orden del dataset.

### 5.1 Paneles gemelos (hallazgo de este proyecto)

Al comparar las celdas de dos paneles **posición por posición**, cinco pares se parecen mucho más que el resto (`notebooks/01_EDA.ipynb`, §7.1):

| Par | r misma posición | r otra posición | Lectura |
|---|---|---|---|
| P14 = P22 | 0,90 | 0,10 | Inequívoco: mismo panel físico fotografiado dos veces |
| M15 = M16 | 0,90 | 0,51 | Muy claro (53 de 60 etiquetas iguales) |
| M11 = M12 | 0,89 | 0,78 | Probable (las celdas mono se parecen mucho entre sí en general) |
| M05 = M08 | 0,83 | 0,61 | Probable |
| M13 = M14 | 0,73 | 0,42 | Probable; solo 12 de 60 etiquetas iguales (¿antes y después de un daño?) |

El dataset no documenta esto. **Criterio conservador:** los cinco pares se tratan como gemelos (`PANELES_GEMELOS`) y siempre van a la misma parte de la partición.

## 6. Conteos

**4 etiquetas × tipo**

| Etiqueta | mono | poly | total | % |
|---|---|---|---|---|
| 0 % | 588 | 920 | 1.508 | 57,5 |
| 33 % | 117 | 178 | 295 | 11,2 |
| 67 % | 56 | 50 | **106** | **4,0** |
| 100 % | 313 | 402 | 715 | 27,2 |
| total | 1.074 | 1.550 | 2.624 | 100 |

**2 clases × tipo (objetivo del proyecto)**

| Clase | mono | poly | total | % |
|---|---|---|---|---|
| FUNCIONAL | 705 | 1.098 | 1.803 | 68,7 |
| DEFECTUOSA | 369 | 452 | 821 | 31,3 |

- El desbalance es de 69/31. Un clasificador que diga siempre "funcional" acierta el 69 %, así que se reportan **exactitud balanceada, F1 de la clase defectuosa y AUC**.
- La clase de 67 % es demasiado pequeña: la clasificación en 4 niveles queda como objetivo secundario.
- Por panel, el porcentaje de celdas defectuosas va de **0 % a 93 %**: hay paneles casi sanos y paneles destruidos.

## 7. Calidad de los datos

| Aspecto | Resultado | Implicación |
|---|---|---|
| Faltantes / corruptos | 0 / 0 | Se usan las 2.624 |
| Duplicados exactos | 0 | — |
| Casi duplicados (r > 0,95) | 1.193 pares, 96 % dentro del mismo panel; 95 pares tienen distinta clase | **Partición por panel**; ruido de etiqueta → techo de exactitud |
| Celdas totalmente apagadas (brillo medio < 40) | 11, todas poly y todas con etiqueta 100 % | Caso trivial; se reportan aparte |
| Saturación | 127 celdas con algún píxel en 255; 15 con > 1 % | Poco frecuente; puede esconder defectos |
| Brillo medio | mín 14 · p5 102 · mediana 156 · p95 192 · máx 228 | Depende del panel → **normalizar cada celda** |
| Busbars por celda | 3 en la gran mayoría; 2 en ~359 poly | **Detectarlos por celda**, no usar una máscara fija |

## 8. Limitaciones y sesgos conocidos

- **Etiquetas subjetivas:** hay un solo experto, sin medición de potencia, y pares casi idénticos con etiquetas distintas. En la literatura se reporta un techo de ~88 % de exactitud (Deitsch et al. 2019, con partición al azar).
- **Pocos paneles:** son 44, de una sola fuente y con una sola cámara. El modelo puede no generalizar a otras cámaras o tecnologías (PERC, half-cut, bifaciales).
- **Reconstrucción de paneles aproximada:** las fronteras de P13–P17 no son exactas y los gemelos son una inferencia. Ambas incertidumbres se resuelven del lado conservador, agrupando.
- **Sin ubicación del defecto:** no hay máscaras de verdad. El mapa de defectos del proyecto solo se puede validar cualitativamente o por su correlación con la etiqueta.

## 9. Diccionario de datos

`data/metadatos.csv` (una fila por celda; la genera `scripts/00_crear_particion.py`):

| Columna | Tipo | Descripción |
|---|---|---|
| `idx` | int | Índice 0…2623 (`cellNNNN.png` con NNNN = idx + 1) |
| `archivo` | str | Ruta relativa dentro del paquete (`images/cell0001.png`) |
| `prob` | float | Probabilidad de defecto original (0, 0,333, 0,667, 1) |
| `tipo` | str | `mono` / `poly` |
| `clase4` | int | 0, 1, 2, 3 (= `prob` × 3) |
| `defectuosa` | int | 1 si `prob` ≥ 0,5 |
| `clase2` | str | `FUNCIONAL` / `DEFECTUOSA` |
| `panel` | int | Panel reconstruido 0…43, en orden |
| `panel_cod` | str | `M01…M18`, `P01…P26` |
| `panel_exacto` | bool | False para P13–P17 (frontera no exacta) |
| `pos_en_panel` | int | Posición de la celda dentro de su panel |
| `grupo` | str | Unidad indivisible para partir (panel, gemelos unidos, bloque ambiguo) |
| `brillo_medio`, `brillo_p05`, `brillo_p95` | float | Estadísticas de intensidad de la celda |
| `frac_saturada` | float | Fracción de píxeles en 255 |
| `oscura` | int | 1 si `brillo_medio` < 40 |

`data/particion.csv`: `idx, archivo, tipo, panel_cod, grupo, prob, clase4, defectuosa, clase2, split` (ver `03_particion.md`).
