#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# G10 — Pipeline complet : Baseline → Grid → Optuna → Landscape
# Protocole P02 : Régularisation & Généralisation
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/pipeline_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════"
echo "  G10 — Pipeline complet P02"
echo "  Timestamp : ${TIMESTAMP}"
echo "═══════════════════════════════════════════════════"

mkdir -p results/figures results/models

# ── Étape 1 : Baseline ──────────────────────────────────────────
echo ""
echo "ÉTAPE 1/4 : Entraînement Baseline"
echo "──────────────────────────────────"
poetry run g10-train --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

# ── Étape 2 : Grid Search P02 ───────────────────────────────────
echo ""
echo "ÉTAPE 2/4 : Grid Search P02 (12 configurations)"
echo "─────────────────────────────────────────────────"
poetry run g10-optimize --method grid --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

# ── Étape 3 : Optimisation Bayésienne ──────────────────────────
echo ""
echo "ÉTAPE 3/4 : Optimisation Bayésienne Optuna/TPE (20 trials)"
echo "────────────────────────────────────────────────────────────"
poetry run g10-optimize --method optuna --n-trials 20 --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

# ── Étape 4 : Loss Landscape ────────────────────────────────────
echo ""
echo "ÉTAPE 4/4 : Analyse du Loss Landscape"
echo "───────────────────────────────────────"
poetry run g10-landscape --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

# ── Synthèse ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Pipeline terminé"
echo "  Résultats dans : results/"
echo "  Log            : ${LOG_FILE}"
echo "═══════════════════════════════════════════════════"
