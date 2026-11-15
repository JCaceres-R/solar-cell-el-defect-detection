# Plan de trabajo y avances semanales

Base: el informe de propuesta aprobado, [`Informe_Propuesta_Celdas_EL.pdf`](../Informe_Propuesta_Celdas_EL.pdf) (10-nov-2026). Su cronograma (tabla 6) fija **cinco semanas desde la aprobación**. Este plan lo convierte en cuatro avances semanales y una sustentación final, cada uno con informe y presentación.

> Las semanas son de trabajo. Cada avance se arma **cuando se pide**, con lo que haya hasta ese momento.

**Lo que dice la propuesta y no se cambia sin avisar:**
- Lo único que se evalúa con métricas es el **veredicto**: funcional o defectuosa. El mapa y las medidas son **apoyo visual** y **no se miden** (§5.2 y §7.3), porque ELPV no dice dónde está el defecto.
- El notch y el cierre se comparan con **4 criterios** (tabla 3). **Decide el AUC de la CNN** (criterio 3); los otros tres son de apoyo.

## 1. Calendario

| Semana | Fechas | Tema (tabla 6) | Cierra con |
|---|---|---|---|
| 1 | 9 – 15 nov | Datos: análisis, paneles, partición, máscara, normalización, GPU | **Avance 1** |
| 2 | 16 – 22 nov | Busbars: notch y cierre; comparación con las métricas de imagen | **Avance 2** |
| 3 | 23 – 29 nov | Mapas y CNN: con A, con B y sin preprocesamiento; elección y congelación | **Avance 3** |
| 4 | 30 nov – 6 dic | Prueba: medición única, análisis de errores y demo | **Avance 4** |
| 5 | 7 – 13 dic | Cierre: informe y sustentación | **Sustentación** |

## 2. Qué se entrega cada semana

| Pieza | Formato | Extensión |
|---|---|---|
| Informe de avance | PDF desde LaTeX (`entregas/avance_N/informe.tex`) | 4 – 6 páginas |
| Presentación | Beamer con el estilo de la propuesta (`entregas/avance_N/presentacion.tex`) | 8 – 12 diapositivas, ~10 min |
| Código | Etiqueta git `avance-N` en el commit que genera las figuras y métricas | — |

### Estructura fija del informe y de la presentación

1. **Resumen**: qué se hizo, el resultado principal con su número y la decisión que se tomó.
2. **Avance frente al cronograma**: tabla con ✅ / 🟡 / ⬜ por actividad de la tabla 6.
3. **Métodos de la semana**: ecuaciones, parámetros elegidos y de dónde salen.
4. **Resultados**: figuras y tablas, medidos en **entrenamiento y validación**. Mono y poly por separado.
5. **Decisiones y cambios respecto a la propuesta**, con su razón.
6. **Problemas y riesgos**.
7. **Plan de la semana siguiente**.

### Reglas para que sea visual y verificable

- Toda figura o número del informe **sale de un script** del repo y se guarda en `reports/figures/` o `reports/metricas/`.
- Prefijos de figuras: `e` EDA · `p` partición · `f` pipeline · `n` normalización y máscara · `b` busbars · `m` mapas · `c` CNN · `t` prueba (test).
- Cada semana muestra al menos una figura **antes / después** de la etapa trabajada y una tabla de métricas.
- **El conjunto de prueba no se toca hasta la semana 4.**

## 3. Cómo se arma un avance cuando se pide

1. **Corte**: merge a `main` de lo que funcione; se regeneran figuras y métricas desde los scripts; etiqueta `avance-N`.
2. **Redacción**: el editor del avance arma el informe y las diapositivas con las 7 secciones fijas.
3. **Revisión**: los otros dos leen y se ensaya una vez (~10 min).

## 4. Plan semana a semana

### Semana 1 · Datos → Avance 1 ✅

| Actividad | Estado |
|---|---|
| Dataset, EDA y ficha de datos (`docs/01_dataset.md`, `docs/02_eda.md`) | ✅ |
| Reconstrucción de 44 paneles y 5 pares de gemelos | ✅ |
| Partición por panel congelada y control de fuga (0 pares cruzados frente a 498 al azar) | ✅ |
| Máscara v1 (respaldo de plantilla) y normalización medidas (`scripts/05_medir_mascara_normalizacion.py`) | ✅ |
| Línea base (características + logística) medida en validación por tipo | ✅ |
| GPU: GTX 1650 4 GB con `torch` 2.14 + CUDA; prueba corta de la CNN (~7 s/época) | ✅ |

Detalle en [`entregas/avance_1/`](../entregas/avance_1/).

### Semana 2 · Busbars → Avance 2

| Actividad | Detalle |
|---|---|
| Detección de busbars | Perfil por filas de Iₙ: número (2 o 3), posición y grosor. Exigir banda fina y de todo el ancho (falla de la línea base con manchas grandes) |
| **A · Notch gaussiano** | Ec. 2 de la propuesta: agujeros gaussianos en ±k·v₀ sobre el eje vertical, sin tocar DC. Dos variantes: **A-global** y **A-filas** |
| **B · Cierre** | Ec. 3: elemento vertical de altura h > w_b, solo en filas de busbar |
| Dedos | Cierre horizontal corto; **revisar las grietas verticales finas**: respuesta de crestas antes y después |
| Comparación | Criterios de imagen de la tabla 3: **1** brillo residual en filas de busbar, **2** cambio fuera de esas filas, **4** tiempo por celda. En validación, mono y poly |

**Figuras:** `b1` espectro con los picos y H(u,v) · `b2` A-global, A-filas y B lado a lado · `b3` perfil por fila antes y después · `b4` grietas verticales antes y después de quitar dedos.
**Mensaje:** "las dos técnicas quitan los busbars; esto es lo que cada una le hace al resto de la celda".

### Semana 3 · Mapas y CNN → Avance 3

| Actividad | Detalle |
|---|---|
| Mapa de grietas | Sato a escalas finas (ec. 4); en poly, si responde a los granos, Gabor y filtrado por orientación y longitud |
| Mapa de zonas apagadas | Déficit D (ec. 5) y umbral → área apagada |
| **CNN** | 3 canales a 128 × 128 (Ic, D, crestas), pesos por clase, sin rotaciones de 90°. Con **A**, con **B** y con la **imagen original**, 3 semillas cada una |
| **Decisión** | **Criterio 3: AUC de la CNN en validación** decide entre A y B → **sistema congelado** (`sistema-congelado`) |

**Figuras:** `m1` mapas superpuestos (apoyo visual) · `c1` curvas de entrenamiento · `c2` ROC en validación A / B / original · `c3` tabla de los 4 criterios.
**Mensaje:** "el preprocesamiento aporta X puntos de AUC; elegimos A o B por el criterio 3 y congelamos".

### Semana 4 · Prueba → Avance 4

| Actividad | Detalle |
|---|---|
| **Medición única en prueba** | Matriz de confusión, AUC, F1 de defectuosa y exactitud balanceada, por mono y poly |
| Análisis de errores | Falsos positivos y negativos con sus mapas: qué etapa falla en cada uno |
| Aplicación | Recibe una celda con su tipo y muestra veredicto, etapas y mapa. Celdas de los paneles de prueba |
| Si el tiempo alcanza | Cuatro niveles de etiqueta · resumen por panel · inferencia en Hailo |

### Semana 5 · Cierre → Sustentación

- Informe final, que reúne las secciones de los cuatro avances.
- Presentación final y demo en vivo, con video de respaldo.
- Reproducibilidad desde un clon limpio siguiendo el README.

## 5. Responsables

| Línea | Semanas fuertes | Responsable |
|---|---|---|
| Preprocesamiento y busbars (A/B, dedos) | 1 – 2 | _por asignar_ |
| Mapas de grietas y zonas apagadas | 3 | _por asignar_ |
| CNN, GPU, prueba y demo | 3 – 4 | _por asignar_ |
| Editor del avance | rota: S1 · S2 · S3 · S4 | _por asignar_ |

## 6. Riesgos (tabla 5 de la propuesta, más lo visto en la semana 1)

| Riesgo | Qué haremos |
|---|---|
| Sato confunde bordes de grano con grietas en poly | Gabor y filtrado por orientación y longitud |
| El cierre de dedos borra grietas verticales finas | Revisión aparte, antes y después del cierre |
| El notch deja *ringing* | Medirlo también limitado a las filas de los busbars |
| Celdas con 2 busbars en lugar de 3 | Detección por perfil de filas, filtro por celda |
| **El brillo crudo es un atajo de exposición del panel** (semana 1) | La CNN solo ve imágenes normalizadas; la línea base se reporta con y sin esa variable |
| La CNN no supera a la línea base | Es un resultado válido de la comparación |
| Alguien mira la prueba antes de tiempo | `test` solo se carga en la semana 4 |
