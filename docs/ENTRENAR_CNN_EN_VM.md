# Semana 3 · Entrenar la CNN v2 en la VM

La CNN de la semana 3 no se entrena en esta máquina, sino en la VM Linux (Ubuntu 22.04, Python 3.10, 8 CPU, sin GPU).
La VM clona **este** repo en una carpeta aparte. El repo del VAE no se toca. Al final se borra esa carpeta de la VM.

## 1. Qué cambió (27-nov)

| Archivo | Cambio |
|---|---|
| `scripts/12_cnn.py` | **Red v2, más densa:** 5 bloques 32-64-128-192-256 y cabeza 256 → 64 → 1, con 1,89 M de parámetros (la v1 tenía 0,53 M). **5 semillas** (0-4) en vez de 3. `todas` se puede reanudar: salta las corridas que ya tienen su JSON de la v2. Cada JSON guarda la versión de la red, el dispositivo y los hilos. |
| `scripts/13_comparar_cnn.py` | Exige las 30 corridas (6 variantes × 5 semillas), todas de la red v2. Nunca mezcla corridas de la v1 con la v2. La regla de decisión es la misma, con el promedio de 5 semillas. |
| `scripts/vm/run_cnn_vm.sh` | Script nuevo para la VM. Crea el entorno, genera los tensores, corre las 30 corridas, la comparación y la decisión, y copia los pesos del sistema elegido a `modelos/cnn_v2/`. |
| `requirements-vm.txt` | Versiones para Python 3.10. Las de numpy, pandas, scipy, sklearn, joblib, matplotlib y pillow son las mismas que ya funcionan en la VM del VAE. |
| `.gitignore`, `.gitattributes` | `.venv-linux/` no se versiona. Los pesos de `modelos/cnn_v2/` sí, porque la semana 4 los necesita. Los `.sh` siempre con fin de línea LF. |

Las 5 corridas de la v1 que se hicieron aquí (solo con la semilla 0) se movieron a `data/interim/cnn_v1_local/`, que no
se versiona. No sirven para comparar con la v2.

**Comprobado aquí, con una prueba de humo** (1 época en la GPU y 30 resultados falsos, todo borrado después):
`12_cnn.py` escribe el JSON con `red: v2`, `todas` salta las corridas hechas, `13_comparar_cnn.py` llega hasta la
decisión y genera c1-c3, y `bash -n` acepta el script de la VM. **Lo que no se pudo probar aquí:** la instalación en
Python 3.10 y la velocidad real de la VM.

## 2. Cuánto va a tardar

Medido en esta máquina con la CPU (4 hilos), un lote de 32 tarda 1,5 s con la v2. Eso da ~1,4 min por época solo de
entrenamiento y ~1,8 min con la validación. Con 40 épocas, **1 a 1,5 h por corrida**, y las 30 corridas suman
**unas 30 a 45 h** si la VM rinde parecido con sus 8 CPU.

- El script anota en `reports/vm_cnn/RUN_LOG.md` el tiempo de cada corrida y en la consola muestra cuánto falta.
  **Revisar ese tiempo después de la primera corrida.**
- Empieza con las instrucciones SIMD completas, que son más rápidas. Si aparece un SIGILL (exit 132) como los que hubo
  con el VAE, reintenta esa corrida y sigue con el conjunto limitado de instrucciones, que es más lento pero estable.
- Recorre las corridas por semilla: primero las 6 variantes con la semilla 0, luego con la 1, etc. Si hace falta mostrar
  algo antes de terminar, ya hay comparaciones con las primeras semillas.

## 3. Pasos

### En esta máquina: subir la semana 3 a GitHub

La VM clona desde GitHub, así que primero hay que subir lo de la semana 3 (los scripts 10-13, `src/mapas.py`, las
figuras m1-m3 y los archivos de la VM). Commit sugerido:

```
git add .gitattributes .gitignore requirements-vm.txt src/mapas.py scripts/10_mapas.py scripts/11_tensores_cnn.py \
        scripts/12_cnn.py scripts/13_comparar_cnn.py scripts/vm/run_cnn_vm.sh docs/ENTRENAR_CNN_EN_VM.md \
        docs/plan_semanal.md reports/figures/m1_mapas.png reports/figures/m2_poly_granos.png \
        reports/figures/m3_barrido_mapas.png reports/metricas/semana3_mapas.json
git commit -m "Semana 3: mapas, tensores y CNN v2 (5 semillas) lista para entrenar en la VM"
git push
```

### En la VM

```bash
cd ~
git clone https://github.com/JCaceres-R/solar-cell-el-defect-detection.git solar-cell
cd solar-cell
tmux new -A -s cnn
bash scripts/vm/run_cnn_vm.sh
# Ctrl+B D para salir sin cortarlo; tmux attach -t cnn para volver
```

- Si el repo es privado, `git clone` pide usuario y token: usa el mismo token con el que la VM hace push al repo del VAE.
- Si se corta (por ejemplo, se reinicia la VM), se vuelve a lanzar `bash scripts/vm/run_cnn_vm.sh`. No repite los
  tensores ni las corridas que ya terminaron. La corrida que estaba en curso sí empieza de cero.
- Para ver cómo va desde otra terminal: `tail -f reports/vm_cnn/logs/E20_*_s*.txt` o `cat reports/vm_cnn/RUN_LOG.md`.

### En la VM, al terminar: subir los resultados

El script imprime los comandos al final:

```bash
git add reports/metricas/cnn reports/metricas/semana3_cnn.json reports/figures/c1_entrenamiento.png \
        reports/figures/c2_roc_val.png reports/figures/c3_criterios.png reports/vm_cnn modelos/cnn_v2
git commit -m "Semana 3: 30 corridas de la CNN v2 en la VM y decisión entre A y B"
git push
```

### En la VM: borrar solar-cell

Solo después de comprobar en GitHub que el push llegó, es decir, que se ven `reports/metricas/semana3_cnn.json` y
`modelos/cnn_v2/`:

```bash
tmux kill-session -t cnn
rm -rf ~/solar-cell
```

Con eso se borran el código, el entorno `.venv-linux`, los tensores y los pesos de las 30 corridas.

## 4. Qué revisar cuando vuelva

1. `git pull` en esta máquina.
2. `reports/vm_cnn/RUN_LOG.md`: las 30 corridas en OK, el tiempo de cada una y si hubo algún SIGILL.
3. `reports/metricas/semana3_cnn.json` → `decision`: método elegido (A-global, A-filas o B), si se quitan los dedos y
   el motivo.
4. Las figuras `c1_entrenamiento.png`, `c2_roc_val.png` y `c3_criterios.png`, que van al Avance 3.

**Semana 4:** los pesos del sistema elegido (5 semillas) quedan en `modelos/cnn_v2/`. La medición en prueba es solo
inferencia sobre 402 celdas, así que se puede correr en esta máquina sin entrenar nada.
