"""
Grid Search P02 — Exploration exhaustive weight_decay × dropout.

Protocole : 4 valeurs de weight_decay × 3 valeurs de dropout = 12 configurations.
Le learning rate est fixé à 2e-5 pour isoler l'effet des régulateurs.

Résultat attendu : DataFrame avec métriques par configuration,
exporté en CSV pour analyse comparative.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from g10_camembert.models.camembert import load_camembert, get_device
from g10_camembert.training.trainer import train_model, TrainingResult
from g10_camembert.utils.metrics import generalization_gap


# Grille du protocole P02
WEIGHT_DECAY_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
DROPOUT_GRID = [0.0, 0.1, 0.3]


def run_grid_search(
    train_dataset: Dataset,
    val_dataset: Dataset,
    weight_decay_grid: list[float] = WEIGHT_DECAY_GRID,
    dropout_grid: list[float] = DROPOUT_GRID,
    lr: float = 2e-5,
    num_epochs: int = 2,
    batch_size: int = 16,
    grad_accum: int = 2,
    seed: int = 42,
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Exécute le grid search exhaustif du protocole P02.

    Args:
        train_dataset:    Dataset d'entraînement PyTorch.
        val_dataset:      Dataset de validation PyTorch.
        weight_decay_grid: Valeurs de weight_decay à tester.
        dropout_grid:     Valeurs de dropout à tester.
        lr:               Learning rate fixé pour isolation.
        num_epochs:       Époques par configuration.
        batch_size:       Taille de mini-batch.
        grad_accum:       Accumulation de gradient.
        seed:             Graine aléatoire.
        results_dir:      Dossier de sauvegarde CSV.

    Returns:
        Tuple (df_grid, grid_histories) :
        - df_grid : DataFrame avec colonnes [weight_decay, dropout,
          train_f1, val_f1, gap, time_s].
        - grid_histories : dict label → historique d'entraînement.
    """
    device = get_device()
    grid_results = []
    grid_histories: dict[str, dict] = {}
    total = len(weight_decay_grid) * len(dropout_grid)
    current = 0

    logger.info(f"Grid Search P02 : {total} configurations à tester")
    logger.info(f"  weight_decay ∈ {weight_decay_grid}")
    logger.info(f"  dropout ∈ {dropout_grid}")

    for wd in weight_decay_grid:
        for dp in dropout_grid:
            current += 1
            label = f"wd={wd:.0e}_dp={dp}"
            logger.info(f"\n[{current}/{total}] weight_decay={wd:.0e} | dropout={dp}")

            model = load_camembert(dropout=dp, device=device)
            result: TrainingResult = train_model(
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
                seed=seed,
                verbose=True,
            )

            hist = result.history
            train_f1 = max(hist.train_f1) if hist.train_f1 else 0.0
            val_f1 = result.best_val_f1
            gap_info = generalization_gap(train_f1, val_f1)

            grid_results.append({
                "weight_decay": wd,
                "dropout": dp,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "gap": gap_info["gap"],
                "time_s": result.time_s,
            })
            grid_histories[label] = {
                "train_f1": hist.train_f1,
                "val_f1": hist.val_f1,
                "train_loss": hist.train_loss,
                "val_loss": hist.val_loss,
            }

            del model

    df_grid = pd.DataFrame(grid_results)
    df_grid = df_grid.sort_values("val_f1", ascending=False).reset_index(drop=True)

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "grid_p02_results.csv"
        df_grid.to_csv(csv_path, index=False)
        logger.info(f"✅ Résultats grid sauvegardés : {csv_path}")

    logger.info(f"\n✅ Grid Search terminé")
    logger.info(f"  Meilleure config : {df_grid.iloc[0].to_dict()}")

    return df_grid, grid_histories


def main() -> None:
    """Point d'entrée CLI pour le grid search."""
    import typer
    from omegaconf import OmegaConf
    from g10_camembert.data import load_allocine, prepare_splits, AllocinéDataset
    from g10_camembert.models.camembert import load_tokenizer

    app = typer.Typer()

    @app.command()
    def run(config: str = typer.Option("configs/config.yaml")) -> None:
        cfg = OmegaConf.load(config)
        dataset = load_allocine()
        train_s, val_s, _ = prepare_splits(
            dataset,
            n_train=cfg.dataset.n_train_per_class,
            n_val=cfg.dataset.n_val_per_class,
            seed=cfg.project.seed,
        )
        tokenizer = load_tokenizer(cfg.model.name)
        train_ds = AllocinéDataset(train_s, tokenizer, cfg.dataset.max_seq_len)
        val_ds = AllocinéDataset(val_s, tokenizer, cfg.dataset.max_seq_len)

        df, histories = run_grid_search(
            train_ds,
            val_ds,
            weight_decay_grid=list(cfg.protocol_p02.weight_decay_grid),
            dropout_grid=list(cfg.protocol_p02.dropout_grid),
            lr=cfg.protocol_p02.grid_lr,
            num_epochs=cfg.protocol_p02.grid_num_epochs,
            results_dir=Path(cfg.project.results_dir),
            seed=cfg.project.seed,
        )
        print(df.to_string(index=False))

    app()


if __name__ == "__main__":
    main()
