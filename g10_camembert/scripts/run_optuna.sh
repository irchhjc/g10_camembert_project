#!/usr/bin/env bash
# Optimisation Bayésienne Optuna/TPE
set -euo pipefail
N_TRIALS=${1:-20}
echo "Lancement Optuna (${N_TRIALS} trials TPE)..."
poetry run g10-optimize --method optuna --n-trials "${N_TRIALS}" --config configs/config.py
echo "✅ Optuna terminé. Meilleurs params : results/best_params.json"
