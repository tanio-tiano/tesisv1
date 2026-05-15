#!/usr/bin/env bash
# Runner Linux/macOS para la comparacion WO_base vs WO_shap sobre la instancia grande.
# Llama al runner unificado runners/compare_base_vs_shap.py (parametrizado por
# --problem tmlap:<path>) con el patch --init-mode random.
#
# Uso:
#   bash run.sh              # 1 corrida (default)
#   RUNS=30 bash run.sh      # override numero de corridas via env var
#   bash run.sh --runs 5     # override via argumento pasado a python
#
# Salidas: experiments/tmlap_grande/outputs/run_<RUNS>r/
set -euo pipefail

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$THIS_DIR/../.." && pwd )"

RUNS="${RUNS:-1}"
AGENTS="${AGENTS:-30}"
ITERATIONS="${ITERATIONS:-300}"
SEED="${SEED:-1234}"
PROFILE="${PROFILE:-soft}"
SHAPLEY_STEPS="${SHAPLEY_STEPS:-3}"

if [ ! -f "$THIS_DIR/4.instancia_grande.txt" ]; then
    echo ">> Generando 4.instancia_grande.txt (no existe)..."
    cd "$THIS_DIR"
    python generate_instance.py
    cd "$PROJECT_ROOT"
fi

OUTPUT_DIR="$THIS_DIR/outputs/run_${RUNS}r_${PROFILE}"
echo ">> WO_base vs WO_shap sobre tmlap:4.instancia_grande.txt"
echo "   runs=$RUNS agents=$AGENTS iterations=$ITERATIONS seed=$SEED profile=$PROFILE shapley_steps=$SHAPLEY_STEPS"
echo "   output=$OUTPUT_DIR"

cd "$PROJECT_ROOT"
python -m runners.compare_base_vs_shap \
    --problem "tmlap:experiments/tmlap_grande/4.instancia_grande.txt" \
    --runs "$RUNS" \
    --agents "$AGENTS" \
    --iterations "$ITERATIONS" \
    --seed "$SEED" \
    --profile "$PROFILE" \
    --shapley-steps "$SHAPLEY_STEPS" \
    --init-mode random \
    --output "$OUTPUT_DIR" \
    "$@"

echo ">> Listo. Resultados en $OUTPUT_DIR"
