#!/usr/bin/env python
"""
Optimisation Bayésienne Optuna/TPE — Protocole P02.
Usage : poetry run python run_optuna.py [--n-trials N]
"""
import sys
from pathlib import Path

from configs.config import CFG
from g10_camembert.loader import load_allocine, balanced_subsample
from g10_camembert.dataset import AllocinéDataset
from g10_camembert.camembert import load_tokenizer
from g10_camembert.optuna_search import run_optuna_study

if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else CFG.optuna.n_trials

    dataset = load_allocine()
    tokenizer = load_tokenizer(CFG.model.name)

    train_s = balanced_subsample(
        dataset["train"], CFG.dataset.n_optuna_train_per_class, seed=CFG.project.seed
    )
    val_s = balanced_subsample(
        dataset["validation"], CFG.dataset.n_optuna_val_per_class, seed=CFG.project.seed
    )
    train_ds = AllocinéDataset(train_s, tokenizer, CFG.dataset.max_seq_len)
    val_ds   = AllocinéDataset(val_s,   tokenizer, CFG.dataset.max_seq_len)

    study = run_optuna_study(
        train_ds, val_ds,
        study_name=CFG.optuna.study_name,
        n_trials=n_trials,
        seed=CFG.project.seed,
        results_dir=Path(CFG.project.results_dir),
    )
    print(f"\nMeilleur F1-val : {study.best_value:.4f}")
    print(f"Meilleurs params : {study.best_params}")
