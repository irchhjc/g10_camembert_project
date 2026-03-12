#!/usr/bin/env bash
# Entraînement baseline (hyperparamètres par défaut)
set -euo pipefail
echo "Lancement entraînement baseline..."
poetry run g10-train --config configs/config.yaml
echo "✅ Baseline terminé."
