#!/usr/bin/env bash
# Barrido del protocolo sobre CEC2022 (todas las funciones F1..F12): corre los
# 4 presupuestos MaxFES {5e3, 5e4, 5e5, 5e6} en secuencia, RUNS corridas c/u.
# Se auto-lanza en segundo plano (nohup) y sobrevive a la desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./run_remote_cec.sh [--runs N] [--dim N] [--agents N] [--budgets "5000,50000,500000"]
#   ./run_remote_cec.sh [RUNS] [DIM] [AGENTS]                  (legacy posicional)
#
# Opciones:
#   -r, --runs N         Cantidad de corridas por (problema, MaxFES) (default: 5)
#   -d, --dim N          Dimension del problema (default: 10; protocolo usa 10 y 20)
#   -a, --agents N       Tamano de poblacion (default: 30)
#   -b, --budgets LISTA  Lista separada por coma de MaxFES a barrer
#                        (default: 5000,50000,500000,5000000)
#   -h, --help           Muestra esta ayuda
#
# Ejemplos:
#   ./run_remote_cec.sh                                       # 5 runs, dim=10, 4 MaxFES
#   ./run_remote_cec.sh --runs 51                             # protocolo completo (51 runs)
#   ./run_remote_cec.sh --runs 10 --dim 20                    # dim=20 con 10 runs
#   ./run_remote_cec.sh --runs 5 --budgets "5000,50000"       # solo 2 MaxFES
#   ./run_remote_cec.sh 51 20                                  # legacy posicional sigue OK
set -euo pipefail

usage() {
  sed -n '2,22p' "$0"
}

RUNS=5
DIM=10
AGENTS=30
BUDGETS_STR="5000,50000,500000,5000000"

# Si el primer argumento empieza con '-', parsear como flags. Si no, modo posicional legacy.
if [[ $# -gt 0 && "$1" == -* ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -r|--runs)        RUNS="${2:?--runs requiere un valor}"; shift 2 ;;
      -d|--dim)         DIM="${2:?--dim requiere un valor}"; shift 2 ;;
      -a|--agents)      AGENTS="${2:?--agents requiere un valor}"; shift 2 ;;
      -b|--budgets)     BUDGETS_STR="${2:?--budgets requiere un valor}"; shift 2 ;;
      -h|--help)        usage; exit 0 ;;
      *) echo "ERROR: flag desconocido: $1" >&2; usage; exit 1 ;;
    esac
  done
else
  RUNS="${1:-5}"
  DIM="${2:-10}"
  AGENTS="${3:-30}"
fi

IFS=',' read -r -a BUDGETS <<< "$BUDGETS_STR"
LOG="ablation_cec2022_d${DIM}_sweep.log"

# 1ra invocacion: chequea opfunu y se re-lanza en background (nohup).
if [ "${CEC_BG:-0}" != "1" ]; then
  if ! python -c "import opfunu" 2>/dev/null; then
    echo "ERROR: falta la libreria 'opfunu' (pip install opfunu)." >&2
    exit 1
  fi
  echo "Config:  runs=$RUNS  dim=$DIM  agents=$AGENTS  budgets=$BUDGETS_STR"
  CEC_BG=1 BUDGETS_STR="$BUDGETS_STR" nohup bash "$0" --runs "$RUNS" --dim "$DIM" --agents "$AGENTS" --budgets "$BUDGETS_STR" > "$LOG" 2>&1 &
  echo "Lanzado PID $!  ->  log: $LOG"
  echo "Salidas:  experiments/cec2022_d${DIM}_fes<MaxFES>/{base,shap}/values/"
  echo "Seguir progreso:  tail -f $LOG"
  exit 0
fi

# --- cuerpo real (corre en segundo plano) ---
echo "##### BARRIDO CEC2022 dim=$DIM agentes=$AGENTS runs=$RUNS budgets=$BUDGETS_STR | inicio $(date) #####"
for FES in "${BUDGETS[@]}"; do
  OUT="experiments/cec2022_d${DIM}_fes${FES}"
  echo "===== CEC2022:all | MaxFES=$FES | salida=$OUT | $(date) ====="
  python -m runners.run_ablation \
    --problem "cec2022:all" --dim "$DIM" --agents "$AGENTS" --max-fes "$FES" \
    --runs "$RUNS" --modes base,shap \
    --output "$OUT" \
    || echo "!!! FALLO en MaxFES=$FES (continuo con el resto) | $(date)"
done
echo "##### BARRIDO COMPLETO | fin $(date) #####"
