#!/usr/bin/env bash
# Lanza el ablation B4 sobre CEC2022 (todas las funciones F1..F12) en segundo
# plano; sobrevive a la desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./run_remote_cec.sh [runs] [max_fes] [dim] [agents] [output_dir]
# Por defecto (lo pedido): 51 runs, MaxFES=5000 (5e3, el menor del protocolo),
# dim=10, 30 agentes, modos base,exact.
# Ejemplos:
#   ./run_remote_cec.sh                 # 51 runs, MaxFES=5000, dim=10, 30 agentes
#   ./run_remote_cec.sh 51 50000        # mismo pero MaxFES=50000
#   ./run_remote_cec.sh 51 500000 20    # MaxFES=5e5, dim=20
set -euo pipefail

RUNS="${1:-51}"
MAXFES="${2:-5000}"
DIM="${3:-10}"
AGENTS="${4:-30}"
OUT="${5:-experiments/cec2022_d${DIM}_fes${MAXFES}}"
LOG="ablation_$(basename "$OUT").log"

# CEC2022 necesita la libreria opfunu (TMLAP no la usaba).
if ! python -c "import opfunu" 2>/dev/null; then
  echo "ERROR: falta la libreria 'opfunu' (pip install opfunu)." >&2
  exit 1
fi

echo "CEC2022:all | runs=$RUNS | MaxFES=$MAXFES | dim=$DIM | agentes=$AGENTS | modos=base,exact"

nohup python experiments/ablation_b4/run_ablation.py \
  --problem "cec2022:all" --dim "$DIM" --agents "$AGENTS" --max-fes "$MAXFES" \
  --runs "$RUNS" --modes base,exact \
  --output "$OUT" \
  > "$LOG" 2>&1 &

echo "Lanzado PID $!  ->  log: $LOG  ->  salida: $OUT/values/"
echo "Seguir progreso:  tail -f $LOG"
