#!/usr/bin/env bash
# Smoke MLPAP: 1 instancia (default S01), 5 corridas, MaxFES=5e5, ambos modos.
# Foreground (feedback rapido). No usa nohup.
#
# Uso (SIEMPRE desde la raiz del repo):
#   ./scripts/run_remote_mlpap_smoke.sh                 # S01, 5 runs
#   ./scripts/run_remote_mlpap_smoke.sh M01             # otra instancia
#   INSTANCES_DIR=/otra/ruta ./scripts/run_remote_mlpap_smoke.sh XL01
#
# Requisitos: numba opcional; sin numba corre en fallback numpy (mas lento).
set -euo pipefail

INSTANCE="${1:-S01}"
INSTANCES_DIR="${INSTANCES_DIR:-data/mlpap}"
RUNS="${RUNS:-5}"
MAXFES="${MAXFES:-500000}"

if [ ! -f "${INSTANCES_DIR}/${INSTANCE}.json" ]; then
  echo "ERROR: no se encuentra ${INSTANCES_DIR}/${INSTANCE}.json" >&2
  echo "Fijar INSTANCES_DIR si el dataset esta fuera del repo." >&2
  exit 1
fi

echo "Config:  instance=${INSTANCE}  runs=${RUNS}  MaxFES=${MAXFES}  dir=${INSTANCES_DIR}"
python -c "import numba" 2>/dev/null && echo "numba: OK (kernel JIT compilado)" || echo "numba: no instalado (fallback numpy)"

python -m runners.run_ablation \
  --problem "mlpap:${INSTANCES_DIR}/${INSTANCE}.json" \
  --agents 30 --max-fes "${MAXFES}" --runs "${RUNS}" \
  --modes base,shap --no-exact-optimum \
  --output "experiments/_smoke_mlpap_${INSTANCE}"

echo "Smoke completo. Salida: experiments/_smoke_mlpap_${INSTANCE}/{base,shap}/values/"
