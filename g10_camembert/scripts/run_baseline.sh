#!/usr/bin/env bash
# Entraînement baseline (hyperparamètres par défaut)
set -euo pipefail
echo "Lancement entraînement baseline..."
poetry run g10-train --config configs/config.py
echo "Baseline terminé."
