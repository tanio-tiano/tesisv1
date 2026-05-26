#!/usr/bin/env bash
# Lanza el ablation B4 (TMLAP dura) en segundo plano; sobrevive a la desconexion SSH.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./run_remote.sh [--runs N] [--max-fes N] [--output PATH]
#   ./run_remote.sh [RUNS] [MAXFES] [OUTPUT_DIR]        (legacy posicional)
#
# Opciones:
#   -r, --runs N         Cantidad de corridas (default: 5)
#   -f, --max-fes N      Presupuesto MaxFES (default: 50000)
#   -o, --output PATH    Directorio salida (default: experiments/ablation_b4_dura_<RUNS>)
#   -h, --help           Muestra esta ayuda
#
# Ejemplos:
#   ./run_remote.sh                                   # 5 runs, MaxFES=50000
#   ./run_remote.sh --runs 10
#   ./run_remote.sh --runs 51 --max-fes 500000        # protocolo completo
#   ./run_remote.sh --runs 5 --max-fes 500000 --output experiments/quick
#   ./run_remote.sh 10 500000                          # legacy posicional sigue funcionando
set -euo pipefail

usage() {
  sed -n '2,18p' "$0"
}

RUNS=5
MAXFES=50000
OUT=""

# Si el primer argumento empieza con '-', parsear como flags. Si no, modo posicional legacy.
if [[ $# -gt 0 && "$1" == -* ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -r|--runs)        RUNS="${2:?--runs requiere un valor}"; shift 2 ;;
      -f|--max-fes)     MAXFES="${2:?--max-fes requiere un valor}"; shift 2 ;;
      -o|--output)      OUT="${2:?--output requiere un valor}"; shift 2 ;;
      -h|--help)        usage; exit 0 ;;
      *) echo "ERROR: flag desconocido: $1" >&2; usage; exit 1 ;;
    esac
  done
else
  RUNS="${1:-5}"
  MAXFES="${2:-50000}"
  OUT="${3:-}"
fi

OUT="${OUT:-experiments/ablation_b4_dura_${RUNS}}"
LOG="ablation_$(basename "$OUT").log"

# Verifica que las instancias se resuelven desde aqui (raiz del repo).
if [ ! -f "3.instancia_dura.txt" ]; then
  echo "ERROR: ejecuta este script desde la raiz del repo (no encuentro 3.instancia_dura.txt)." >&2
  exit 1
fi

echo "Config:  runs=$RUNS  MaxFES=$MAXFES  output=$OUT"

nohup python -m runners.run_ablation \
  --problem "tmlap:3.instancia_dura.txt" --agents 30 --max-fes "$MAXFES" \
  --runs "$RUNS" --modes base,shap \
  --init-mode random --no-exact-optimum --output "$OUT" \
  > "$LOG" 2>&1 &

echo "Lanzado PID $!  ->  log: $LOG  ->  salida: $OUT/{base,shap}/values/"
echo "Seguir progreso:  tail -f $LOG"
