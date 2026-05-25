#!/usr/bin/env bash
# Lanza el ablation B4 (TMLAP dura) en segundo plano; sobrevive a la desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./run_remote.sh [runs] [max_fes] [output_dir]
# Ejemplos:
#   ./run_remote.sh                 # 51 runs, MaxFES=50000
#   ./run_remote.sh 51 500000       # 51 runs, MaxFES=500000
set -euo pipefail

RUNS="${1:-51}"
MAXFES="${2:-50000}"
OUT="${3:-experiments/ablation_b4_dura_${RUNS}}"
LOG="ablation_$(basename "$OUT").log"

# Verifica que las instancias se resuelven desde aqui (raiz del repo).
if [ ! -f "3.instancia_dura.txt" ]; then
  echo "ERROR: ejecuta este script desde la raiz del repo (no encuentro 3.instancia_dura.txt)." >&2
  exit 1
fi

nohup python experiments/ablation_b4/run_ablation.py \
  --problem "tmlap:3.instancia_dura.txt" --agents 30 --max-fes "$MAXFES" \
  --runs "$RUNS" --modes base,blind,shap,wfix --w-fixed 0.5 \
  --init-mode random --output "$OUT" \
  > "$LOG" 2>&1 &

echo "Lanzado PID $!  ->  log: $LOG  ->  salida: $OUT/values/"
echo "Seguir progreso:  tail -f $LOG"
