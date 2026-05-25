#!/usr/bin/env bash
# Barrido del protocolo sobre CEC2022 (todas las funciones F1..F12): corre los
# 4 presupuestos MaxFES {5e3, 5e4, 5e5, 5e6} en secuencia, 51 corridas c/u.
# Se auto-lanza en segundo plano (nohup) y sobrevive a la desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./run_remote_cec.sh [runs] [dim] [agents]
# Por defecto: 51 runs, dim=10, 30 agentes, modos base,exact.
# Ejemplos:
#   ./run_remote_cec.sh            # 51 runs, dim=10, 30 agentes, los 4 MaxFES
#   ./run_remote_cec.sh 51 20      # idem pero dim=20 (el otro dim del protocolo)
set -euo pipefail

RUNS="${1:-51}"
DIM="${2:-10}"
AGENTS="${3:-30}"
BUDGETS=(5000 50000 500000 5000000)   # 5e3, 5e4, 5e5, 5e6 (ascendente)
LOG="ablation_cec2022_d${DIM}_sweep.log"

# 1ra invocacion: chequea opfunu y se re-lanza en background (nohup).
if [ "${CEC_BG:-0}" != "1" ]; then
  if ! python -c "import opfunu" 2>/dev/null; then
    echo "ERROR: falta la libreria 'opfunu' (pip install opfunu)." >&2
    exit 1
  fi
  CEC_BG=1 nohup bash "$0" "$RUNS" "$DIM" "$AGENTS" > "$LOG" 2>&1 &
  echo "Lanzado PID $!  ->  log: $LOG"
  echo "Salidas:  experiments/cec2022_d${DIM}_fes<MaxFES>/values/"
  echo "Seguir progreso:  tail -f $LOG"
  exit 0
fi

# --- cuerpo real (corre en segundo plano) ---
echo "##### BARRIDO CEC2022 dim=$DIM agentes=$AGENTS runs=$RUNS | inicio $(date) #####"
for FES in "${BUDGETS[@]}"; do
  OUT="experiments/cec2022_d${DIM}_fes${FES}"
  echo "===== CEC2022:all | MaxFES=$FES | salida=$OUT | $(date) ====="
  python experiments/ablation_b4/run_ablation.py \
    --problem "cec2022:all" --dim "$DIM" --agents "$AGENTS" --max-fes "$FES" \
    --runs "$RUNS" --modes base,exact \
    --output "$OUT" \
    || echo "!!! FALLO en MaxFES=$FES (continuo con el resto) | $(date)"
done
echo "##### BARRIDO COMPLETO | fin $(date) #####"
