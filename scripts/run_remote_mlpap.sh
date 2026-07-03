#!/usr/bin/env bash
# Barrido MLPAP completo: 100 instancias (S01-2XL20), MaxFES=5e5, 30 corridas
# c/u, modos base,shap. Se auto-lanza en background (nohup) y sobrevive a la
# desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./scripts/run_remote_mlpap.sh                        # defaults
#   RUNS=30 MAXFES=500000 ./scripts/run_remote_mlpap.sh
#   INSTANCES_DIR=/ruta/completa ./scripts/run_remote_mlpap.sh
#
# Variables de entorno:
#   RUNS           corridas por instancia (default: 30)
#   MAXFES         presupuesto por corrida (default: 500000 = 5e5)
#   INSTANCES_DIR  ruta al directorio con S01.json...2XL20.json
#                  (default: experiments/MLAP/Instances)
#   SCALES         lista de escalas a barrer separada por espacios
#                  (default: "S M L XL 2XL")
#   MODES          modos a correr (default: "base,shap"). Para paralelizar por
#                  modo, lanzar DOS procesos: MODES=base y MODES=shap. Las
#                  semillas no dependen del modo -> el pareo se preserva.
#
# Paralelo por modo (2 procesos, mismo OUT, sin colision de carpetas):
#   MODES=base ./scripts/run_remote_mlpap.sh
#   MODES=shap ./scripts/run_remote_mlpap.sh
#
# Requisitos: numba opcional (acelera ~5-10x); sin numba corre en fallback.
set -euo pipefail

RUNS="${RUNS:-30}"
MAXFES="${MAXFES:-500000}"
INSTANCES_DIR="${INSTANCES_DIR:-experiments/MLAP/Instances}"
SCALES="${SCALES:-S M L XL 2XL}"
MODES="${MODES:-base,shap}"

OUT="experiments/mlpap_all_fes${MAXFES}_r${RUNS}"
LOG="ablation_mlpap_$(echo "$MODES" | tr ',' '-').log"

# Enumeracion de instancias segun escalas pedidas.
declare -a INSTS=()
for s in $SCALES; do
  for i in $(seq -w 1 20); do
    INSTS+=("${s}${i}")
  done
done

# 1ra invocacion: valida entorno y se re-lanza en background.
if [ "${MLPAP_BG:-0}" != "1" ]; then
  # Chequeo de que el directorio existe y contiene la primera instancia.
  first="${INSTS[0]}"
  if [ ! -f "${INSTANCES_DIR}/${first}.json" ]; then
    echo "ERROR: no se encuentra ${INSTANCES_DIR}/${first}.json" >&2
    echo "Ajustar INSTANCES_DIR para apuntar al dataset (contiene S01.json...2XL20.json)." >&2
    exit 1
  fi
  # Aviso numba.
  python -c "import numba" 2>/dev/null && NUMBA_MSG="OK" || NUMBA_MSG="NO (fallback numpy, mas lento)"

  echo "Config:  runs=${RUNS}  MaxFES=${MAXFES}  modes=${MODES}  instancias=${#INSTS[@]}  numba=${NUMBA_MSG}"
  echo "Dir instancias: ${INSTANCES_DIR}"
  echo "Salida:         ${OUT}/<instance>/{base,shap}/values/"
  MLPAP_BG=1 RUNS="$RUNS" MAXFES="$MAXFES" INSTANCES_DIR="$INSTANCES_DIR" SCALES="$SCALES" MODES="$MODES" \
    nohup bash "$0" > "$LOG" 2>&1 &
  echo "Lanzado PID $!  ->  log: $LOG"
  echo "Seguir progreso: tail -f $LOG"
  exit 0
fi

# --- cuerpo real (corre en segundo plano) ---
echo "##### BARRIDO MLPAP: ${#INSTS[@]} instancias | inicio $(date) #####"
for INST in "${INSTS[@]}"; do
  OUT_INST="${OUT}/${INST}"
  echo "===== ${INST} | salida=${OUT_INST} | $(date) ====="
  python -m runners.run_ablation \
    --problem "mlpap:${INSTANCES_DIR}/${INST}.json" \
    --agents 30 --max-fes "${MAXFES}" --runs "${RUNS}" \
    --modes "${MODES}" --no-exact-optimum \
    --output "${OUT_INST}" \
    || echo "!!! FALLO en ${INST} | $(date)"
done
echo "##### BARRIDO COMPLETO | fin $(date) #####"
