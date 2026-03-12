"""
Optimisation Bayésienne Optuna/TPE pour le protocole P02.

L'algorithme TPE (Tree-structured Parzen Estimator) explore conjointement :
- weight_decay (catégoriel)
- dropout (catégoriel)
- learning_rate (continu, log-scale)

Le pruner MedianPruner arrête précocement les trials sous-performants.

Référence : Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import optuna
import typer
from loguru import logger
from torch.utils.data import Dataset

from g10_camembert.camembert import get_device, load_camembert
from g10_camembert.trainer import train_model
from g10_camembert.metrics import generalization_gap


# Espace de recherche du protocole P02
WEIGHT_DECAY_CHOICES = [1e-5, 1e-4, 1e-3, 1e-2]
DROPOUT_CHOICES = [0.0, 0.1, 0.3]
LR_MIN = 1e-6
LR_MAX = 5e-4


def create_objective(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 16,
    grad_accum: int = 2,
    num_epochs: int = 2,
    seed: int = 42,
) -> callable:
    """
    Crée la fonction objectif pour Optuna.

    La closure capture les datasets pour éviter de les passer
    à chaque appel de trial.

    Args:
        train_dataset:  Dataset d'entraînement (sous-ensemble Optuna).
        val_dataset:    Dataset de validation (sous-ensemble Optuna).
        batch_size:     Taille de mini-batch.
        grad_accum:     Accumulation de gradient.
        num_epochs:     Époques par trial.
        seed:           Graine de base (chaque trial utilise seed + trial.number).

    Returns:
        Fonction objectif compatible Optuna.
    """
    device = get_device()

    def objective(trial: optuna.Trial) -> float:
        """
        Objectif : maximiser F1-score macro sur la validation.

        Variables explorées :
        - weight_decay : catégoriel (P02 grid)
        - dropout      : catégoriel (P02 grid)
        - learning_rate: continu log-scale
        """
        wd = trial.suggest_categorical("weight_decay", WEIGHT_DECAY_CHOICES)
        dp = trial.suggest_categorical("dropout", DROPOUT_CHOICES)
        lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)

        model = load_camembert(dropout=dp, device=device)
        result = train_model(
            model,
            train_dataset,
            val_dataset,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size,
            grad_accum=grad_accum,
            num_epochs=num_epochs,
            early_stopping_patience=2,
            device=device,
            seed=seed + trial.number,
            verbose=False,
        )

        val_f1 = result.best_val_f1
        train_f1 = max(result.history.train_f1) if result.history.train_f1 else 0.0
        gap = generalization_gap(train_f1, val_f1)

        # Enregistrement des attributs pour l'analyse
        trial.set_user_attr("train_f1", train_f1)
        trial.set_user_attr("gap", gap["gap"])
        trial.set_user_attr("time_s", result.time_s)

        logger.info(
            f"  Trial #{trial.number:02d} | "
            f"wd={wd:.0e} dp={dp:.2f} lr={lr:.2e} → "
            f"F1_val={val_f1:.4f} (gap={gap['gap']:+.4f})"
        )

        del model
        return val_f1

    return objective


def run_optuna_study(
    train_dataset: Dataset,
    val_dataset: Dataset,
    study_name: str = "g10_p02_regularisation",
    n_trials: int = 20,
    batch_size: int = 16,
    grad_accum: int = 2,
    num_epochs: int = 2,
    seed: int = 42,
    results_dir: Path | None = None,
) -> optuna.Study:
    """
    Exécute l'étude d'optimisation Optuna.

    Args:
        train_dataset: Dataset d'entraînement.
        val_dataset:   Dataset de validation.
        study_name:    Nom de l'étude Optuna.
        n_trials:      Nombre de trials.
        batch_size:    Taille de mini-batch.
        grad_accum:    Accumulation de gradient.
        num_epochs:    Époques par trial.
        seed:          Graine aléatoire.
        results_dir:   Dossier de sauvegarde.

    Returns:
        Objet Study Optuna avec tous les résultats.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)

    db_path = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = f"sqlite:///{results_dir}/optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=db_path,
        load_if_exists=True,
    )

    objective_fn = create_objective(
        train_dataset, val_dataset,
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_epochs=num_epochs,
        seed=seed,
    )

    logger.info(f"Démarrage Optuna — {n_trials} trials (TPE Bayésien)...")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"\n{'='*60}")
    logger.info("  RÉSULTATS OPTUNA")
    logger.info(f"{'='*60}")
    logger.info(f"  Meilleur F1-val  : {study.best_value:.4f}")
    logger.info(f"  Meilleurs params :")
    for k, v in study.best_params.items():
        logger.info(f"    {k:20s} = {v}")

    # Sauvegarde des résultats
    if results_dir is not None:
        # Pickle de l'étude complète
        with open(results_dir / "optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)

        # JSON des meilleurs paramètres
        best_params = {
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        with open(results_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"\n✅ Étude sauvegardée dans {results_dir}")

    return study


def _cli_optuna(
    config: str = typer.Option("configs/config.py"),
    method: str = typer.Option("optuna", help="optuna ou grid"),
    n_trials: int = typer.Option(20),
) -> None:
    from g10_camembert.config_loader import load_config
    from g10_camembert.loader import load_allocine, balanced_subsample
    from g10_camembert.dataset import AllocinéDataset
    from g10_camembert.camembert import load_tokenizer

    cfg = load_config(config)
    dataset = load_allocine()
    tokenizer = load_tokenizer(cfg.model.name)

    # Datasets réduits pour Optuna
    train_s = balanced_subsample(
        dataset["train"],
        cfg.dataset.n_optuna_train_per_class,
        seed=cfg.project.seed,
    )
    val_s = balanced_subsample(
        dataset["validation"],
        cfg.dataset.n_optuna_val_per_class,
        seed=cfg.project.seed,
    )
    train_ds = AllocinéDataset(train_s, tokenizer, cfg.dataset.max_seq_len)
    val_ds = AllocinéDataset(val_s, tokenizer, cfg.dataset.max_seq_len)

    study = run_optuna_study(
        train_ds,
        val_ds,
        study_name=cfg.optuna.study_name,
        n_trials=n_trials,
        seed=cfg.project.seed,
        results_dir=Path(cfg.project.results_dir),
    )
    print(f"\nMeilleur F1-val : {study.best_value:.4f}")
    print(f"Meilleurs params : {study.best_params}")


def main() -> None:
    """Point d'entrée CLI pour l'optimisation Optuna."""
    typer.run(_cli_optuna)


if __name__ == "__main__":
    main()
