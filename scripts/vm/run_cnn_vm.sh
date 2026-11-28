#!/usr/bin/env bash
# ============================================================================
# Semana 3 en la VM Linux (solo CPU): tensores de la CNN, 30 entrenamientos de la red v2 (6 variantes × 5 semillas),
# comparación entre A y B (criterio 3) y copia de los pesos del sistema elegido.
#
# Guía completa: docs/ENTRENAR_CNN_EN_VM.md
#
#   git clone https://github.com/JCaceres-R/solar-cell-el-defect-detection.git ~/solar-cell && cd ~/solar-cell
#   tmux new -A -s cnn
#   bash scripts/vm/run_cnn_vm.sh
#   (Ctrl+B D para salir; tmux attach -t cnn para volver)
#
# Es reanudable: si se corta, volver a lanzarlo salta los tensores y las corridas que ya terminaron.
# Registro: reports/vm_cnn/RUN_LOG.md (resumen y tiempos) y reports/vm_cnn/logs/ (salida completa de cada etapa).
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUT="reports/vm_cnn"; LOGS="$OUT/logs"; MD="$OUT/RUN_LOG.md"
VENV=".venv-linux"; PY="$VENV/bin/python"
VARIANTES=(original In A-global A-filas B B-dedos)
SEMILLAS=(0 1 2 3 4)
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
md() { printf '%s\n' "$@" >> "$MD"; }

# Instrucciones SIMD: primero completas (mucho más rápido en CPU). En esta VM hubo SIGILL (exit 132) intermitentes con
# el repo del VAE; si aparece uno, se reintenta esa corrida y TODAS las siguientes con el conjunto limitado de
# instrucciones que resolvió el problema allá (no cambia los resultados, solo la velocidad).
SIMD_SEGURO="${SIMD_SEGURO:-0}"
modo_seguro() {
  export MKL_ENABLE_INSTRUCTIONS=SSE4_2 DNNL_MAX_CPU_ISA=SSE41 MKL_DEBUG_CPU_TYPE=5 ATEN_CPU_CAPABILITY=default
  SIMD_SEGURO=1
}
[ "$SIMD_SEGURO" = "1" ] && modo_seguro

# correr <log> <comando...>: hasta 3 intentos si muere por SIGILL; desde el primer SIGILL pasa a modo seguro.
correr() {
  local log="$1"; shift; local intento=1 rc
  while :; do
    echo "===== [$(ts)] intento ${intento} (SIMD_SEGURO=${SIMD_SEGURO}): $*" >> "$log"
    "$@" >> "$log" 2>&1; rc=$?
    if [ $rc -eq 132 ] && [ $intento -lt 3 ]; then
      md "- SIGILL (exit 132) en \`$*\`, intento ${intento}; se pasa a modo SIMD seguro y se reintenta."
      modo_seguro; intento=$((intento + 1)); continue
    fi
    return $rc
  done
}

{
  echo ""; echo "# Corrida CNN v2 en la VM — $(ts)"; echo ""
  echo "- Host: $(hostname) · CPUs: $(nproc) · memoria: $(free -g | awk '/Mem:/{print $2" GB"}') · disco libre: $(df -h . | awk 'NR==2{print $4}')"
  echo "- Commit: $(git rev-parse --short HEAD) (rama $(git rev-parse --abbrev-ref HEAD))"
} >> "$MD"

# ---------------------------------------------------------------------------- E00 · entorno
echo "[$(ts)] E00 entorno"
command -v python3 >/dev/null || { echo "FATAL: falta python3"; exit 1; }
[ -x "$PY" ] || python3 -m venv "$VENV" >> "$LOGS/E00_entorno.txt" 2>&1 \
  || { echo "FATAL: no se pudo crear el venv (sudo apt-get install -y python3-venv)"; exit 1; }
{
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
} >> "$LOGS/E00_entorno.txt" 2>&1
modo_dep="versiones exactas de requirements-vm.txt"
if ! "$PY" -m pip install -r requirements-vm.txt >> "$LOGS/E00_entorno.txt" 2>&1; then
  modo_dep="SIN versiones exactas (falló requirements-vm.txt; ver logs/E00_entorno.txt)"
  sed -E 's/[=<>~!].*$//' requirements-vm.txt | grep -vE '^\s*(#|$)' > "$OUT/requirements_sin_versiones.txt"
  "$PY" -m pip install -r "$OUT/requirements_sin_versiones.txt" >> "$LOGS/E00_entorno.txt" 2>&1 \
    || { md "- **FATAL:** pip install falló también sin versiones."; echo "FATAL: pip install"; exit 1; }
fi
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
md "- Python: $("$PY" --version 2>&1) · dependencias: ${modo_dep}" \
   "- torch: $("$PY" -c 'import torch; print(torch.__version__, "| hilos:", torch.get_num_threads())' 2>&1)" \
   "- SIMD seguro al empezar: ${SIMD_SEGURO}"

# ---------------------------------------------------------------------------- E10 · tensores (train y val; la prueba no)
FALTAN=0
for v in "${VARIANTES[@]}"; do [ -f "data/interim/X_${v}.npy" ] || FALTAN=1; done
[ -f data/interim/cnn_indices.csv ] || FALTAN=1
if [ $FALTAN -eq 1 ]; then
  echo "[$(ts)] E10 tensores"; t=$(date +%s)
  correr "$LOGS/E10_tensores.txt" "$PY" scripts/11_tensores_cnn.py \
    || { md "- **FATAL:** 11_tensores_cnn.py falló (ver logs/E10_tensores.txt)."; echo "FATAL: tensores"; exit 1; }
  md "- E10 tensores: OK en $(( $(date +%s) - t )) s"
else
  md "- E10 tensores: ya estaban en data/interim/"
fi

# ---------------------------------------------------------------------------- E20 · 30 entrenamientos
md "" "## E20 · entrenamientos (red v2, 40 épocas)" "" "| corrida | estado | minutos | AUC val (todas / mono / poly) |" "|---|---|---|---|"
TOTAL=$(( ${#VARIANTES[@]} * ${#SEMILLAS[@]} )); n=0; hechas_ahora=0; seg_ahora=0; fallos=0
for s in "${SEMILLAS[@]}"; do
  for v in "${VARIANTES[@]}"; do
    n=$((n + 1)); J="reports/metricas/cnn/${v}_s${s}.json"
    if [ -f "$J" ] && grep -q '"red": "v2"' "$J"; then echo "[$(ts)] ${n}/${TOTAL} ${v} s${s}: ya estaba"; continue; fi
    echo "[$(ts)] ${n}/${TOTAL} ${v} s${s}"; t=$(date +%s)
    if correr "$LOGS/E20_${v}_s${s}.txt" "$PY" scripts/12_cnn.py "$v" "$s"; then
      d=$(( $(date +%s) - t )); hechas_ahora=$((hechas_ahora + 1)); seg_ahora=$((seg_ahora + d))
      auc=$("$PY" -c "import json; r=json.load(open('$J'))['val']; print(' / '.join(f\"{r[k]['auc']:.3f}\" for k in ('todas','mono','poly')))")
      md "| ${v} s${s} | OK | $((d / 60)) | ${auc} |"
      quedan=$((TOTAL - n)); eta=$(( seg_ahora / hechas_ahora * quedan / 60 ))
      echo "      OK en $((d / 60)) min · AUC ${auc} · faltan ${quedan} corridas, unos ${eta} min"
    else
      fallos=$((fallos + 1)); md "| ${v} s${s} | **FALLO** | — | ver logs/E20_${v}_s${s}.txt |"
      echo "      FALLO (ver $LOGS/E20_${v}_s${s}.txt)"
    fi
  done
done
[ $fallos -gt 0 ] && { md "" "**${fallos} corridas fallaron.** Volver a lanzar el script las reintenta; no se compara hasta que estén las 30."; echo "Hay ${fallos} fallos; relanzar el script."; exit 1; }

# ---------------------------------------------------------------------------- E30 · comparación y decisión
echo "[$(ts)] E30 comparación"
correr "$LOGS/E30_comparar.txt" "$PY" scripts/13_comparar_cnn.py \
  || { md "- **FALLO** en 13_comparar_cnn.py (ver logs/E30_comparar.txt)."; echo "FALLO: comparación"; exit 1; }
DEC=$("$PY" -c "import json; d=json.load(open('reports/metricas/semana3_cnn.json', encoding='utf-8'))['decision']; print(d['metodo_busbars'], d['quitar_dedos'])")
md "" "## E30 · decisión" "" "- Método de busbars elegido: **${DEC% *}** · quitar dedos: ${DEC#* }" \
   "- Detalle: reports/metricas/semana3_cnn.json · figuras c1-c3 en reports/figures/"

# ---------------------------------------------------------------------------- E40 · pesos del sistema elegido (para la semana 4)
mkdir -p modelos/cnn_v2
for var in ${DEC% *} $([ "${DEC#* }" = "True" ] && echo B-dedos); do
  for s in "${SEMILLAS[@]}"; do cp "data/interim/cnn_${var}_s${s}.pt" modelos/cnn_v2/; done
done
md "- Pesos copiados a modelos/cnn_v2/: $(ls modelos/cnn_v2 | tr '\n' ' ')" "" "Fin: $(ts)"

cat <<EOF

Listo. Para subir los resultados (desde $REPO):

  git add reports/metricas/cnn reports/metricas/semana3_cnn.json reports/figures/c1_entrenamiento.png \\
          reports/figures/c2_roc_val.png reports/figures/c3_criterios.png reports/vm_cnn modelos/cnn_v2
  git commit -m "Semana 3: 30 corridas de la CNN v2 en la VM y decisión entre A y B"
  git push

EOF
