#!/usr/bin/env python
"""
Pipeline complet G10 — P02 : Baseline → Grid Search → Optuna → Landscape.
Usage : poetry run python run_pipeline.py
"""
import subprocess
import sys

PYTHON = sys.executable
STEPS = [
    ("Baseline",       [PYTHON, "run_baseline.py"]),
    ("Grid Search",    [PYTHON, "run_grid_search.py"]),
    ("Optuna",         [PYTHON, "run_optuna.py"]),
    ("Loss Landscape", [PYTHON, "run_landscape.py"]),
]

if __name__ == "__main__":
    for name, cmd in STEPS:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[ERREUR] Étape '{name}' a échoué (code {result.returncode}). Arrêt.")
            sys.exit(result.returncode)
    print("\n✅ Pipeline complet terminé.")
