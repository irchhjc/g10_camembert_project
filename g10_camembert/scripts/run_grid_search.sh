#!/usr/bin/env bash
# Grid Search P02 : weight_decay × dropout (12 configurations)
set -euo pipefail
echo "Lancement Grid Search P02..."
poetry run g10-optimize --method grid --config configs/config.py
echo "✅ Grid Search terminé. Résultats : results/grid_p02_results.csv"
