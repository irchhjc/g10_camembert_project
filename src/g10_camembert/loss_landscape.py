"""
Analyse du paysage de perte (Loss Landscape) — Protocole P02.

Implémente la méthode de perturbation filtre-normalisée de Li et al. (2018)
pour visualiser la géométrie du paysage de perte autour du minimum convergé.

La filter normalization garantit que les perturbations sont comparables
entre couches de tailles différentes, évitant le biais des couches volumineuses.

Référence : Li, H. et al. (2018) — Visualizing the Loss Landscape of Neural Nets.
            NeurIPS 2018.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from g10_camembert.metrics import compute_sharpness


def _filter_normalize_direction(
    params: list[torch.Tensor],
    direction: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Applique la filter normalization à une direction de perturbation.

    Pour chaque filtre (rangée d'un paramètre ≥ 2D) :
        d_i = d_i × (||p_i|| / ||d_i||)

    Cela garantit que la perturbation relative est équivalente
    quelle que soit la taille du paramètre.

    Args:
        params:    Paramètres originaux du modèle.
        direction: Direction aléatoire à normaliser.

    Returns:
        Direction normalisée par filtre.
    """
    normalized = []
    for p, d in zip(params, direction):
        if p.dim() >= 2:
            for i in range(p.shape[0]):
                norm_p = p[i].norm()
                norm_d = d[i].norm()
                if norm_d > 1e-8 and norm_p > 1e-8:
                    d[i] = d[i] * (norm_p / norm_d)
        normalized.append(d)
    return normalized


def compute_loss_landscape_1d(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device | None = None,
    n_points: int = 8,
    epsilon: float = 0.05,
    n_samples: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """
    Calcule le paysage de perte 1D autour du minimum convergé θ*.

    Procédure :
    1. Générer une direction aléatoire d (filtre-normalisée)
    2. Pour α ∈ [-ε, +ε], calculer L(θ* + α·d)
    3. Restaurer θ* après chaque évaluation

    Args:
        model:     Modèle entraîné (point de convergence θ*).
        dataset:   Dataset d'évaluation.
        device:    Device cible.
        n_points:  Nombre de points d'évaluation sur [-ε, +ε].
        epsilon:   Amplitude maximale de perturbation.
        n_samples: Nombre d'exemples pour l'estimation de la loss.

    Returns:
        Tuple (alphas, losses) :
        - alphas : array de n_points valeurs dans [-ε, +ε]
        - losses : loss correspondante à chaque alpha
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Sauvegarde des paramètres originaux θ*
    orig_params = [p.clone().detach() for p in model.parameters()]

    # Génération d'une direction aléatoire filtre-normalisée
    direction = [torch.randn_like(p) for p in orig_params]
    direction = _filter_normalize_direction(orig_params, direction)

    # Chargeur de données limité à n_samples
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    n_batches = max(1, n_samples // 16)
    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        # Perturbation du modèle
        for p, p0, d in zip(model.parameters(), orig_params, direction):
            p.data = p0 + float(alpha) * d.to(device)

        # Estimation de la loss sur n_batches mini-batches
        batch_losses = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                batch_losses.append(outputs.loss.item())

        losses.append(float(np.mean(batch_losses)))

    # Restauration des paramètres originaux θ*
    for p, p0 in zip(model.parameters(), orig_params):
        p.data = p0.clone()

    logger.debug(f"Landscape calculé : α ∈ [{alphas[0]:.3f}, {alphas[-1]:.3f}]")
    return alphas, losses


def analyze_landscape_multiple_configs(
    models_and_labels: list[tuple],
    dataset: Dataset,
    device: torch.device | None = None,
    n_points: int = 8,
    epsilon: float = 0.05,
    results_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    """
    Analyse comparative du landscape pour plusieurs configurations.

    Args:
        models_and_labels: Liste de (model, label, val_f1, dropout).
        dataset:           Dataset de validation.
        device:            Device cible.
        n_points:          Points d'évaluation.
        epsilon:           Amplitude perturbation.
        results_dir:       Dossier de sauvegarde.

    Returns:
        Tuple (landscape_results, sharpness_data).
    """
    landscape_results: dict[str, tuple] = {}
    sharpness_data = []

    for model, label, val_f1, dropout in models_and_labels:
        logger.info(f"  Calcul landscape : {label}...")
        alphas, losses = compute_loss_landscape_1d(
            model, dataset, device=device,
            n_points=n_points, epsilon=epsilon,
        )
        landscape_results[label] = (alphas, losses)

        base_loss = losses[len(losses) // 2]
        sharpness = compute_sharpness(base_loss, losses)
        sharpness_data.append({
            "label": label,
            "sharpness": sharpness,
            "val_f1": val_f1,
            "dropout": dropout,
        })
        logger.info(f"  Sharpness={sharpness:.5f} | F1-val={val_f1:.4f}")

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "sharpness_results.json", "w") as f:
            json.dump(sharpness_data, f, indent=2)
        logger.info(f"✅ Sharpness sauvegardée : {results_dir / 'sharpness_results.json'}")

    return landscape_results, sharpness_data


def _cli_landscape(
    config: str = typer.Option("configs/config.py"),
    checkpoint: str = typer.Option(None, help="Chemin checkpoint .pt (optionnel)"),
) -> None:
    import json as js

    from g10_camembert.utils.config import load_config
    from g10_camembert.data import load_allocine, prepare_splits, AllocinéDataset
    from g10_camembert.models.camembert import load_camembert, load_tokenizer, get_device
    from g10_camembert.training.trainer import train_model

    cfg = load_config(config)
    device = get_device()

    dataset = load_allocine()
    train_s, val_s, _ = prepare_splits(
        dataset, n_train=cfg.dataset.n_train_per_class,
        n_val=cfg.dataset.n_val_per_class, seed=cfg.project.seed,
    )
    tokenizer = load_tokenizer(cfg.model.name)
    train_ds = AllocinéDataset(train_s, tokenizer, cfg.dataset.max_seq_len)
    val_ds = AllocinéDataset(val_s, tokenizer, cfg.dataset.max_seq_len)

    # Lire les meilleurs params si disponibles
    best_params_path = Path(cfg.project.results_dir) / "best_params.json"
    best_wd = cfg.training.weight_decay_baseline
    best_lr = cfg.training.lr_baseline
    if best_params_path.exists():
        with open(best_params_path) as f:
            bp = js.load(f)
        best_wd = bp["best_params"].get("weight_decay", best_wd)
        best_lr = bp["best_params"].get("learning_rate", best_lr)

    models_and_labels = []
    for dp in list(cfg.landscape.dropout_values_to_compare):
        model = load_camembert(dropout=dp, device=device)
        result = train_model(
            model, train_ds, val_ds,
            lr=best_lr, weight_decay=best_wd,
            batch_size=cfg.training.batch_size,
            num_epochs=3, verbose=False,
        )
        models_and_labels.append((
            model, f"dropout={dp:.1f}", result.best_val_f1, dp
        ))

    landscape_results, sharpness = analyze_landscape_multiple_configs(
        models_and_labels, val_ds, device=device,
        n_points=cfg.landscape.n_points,
        epsilon=cfg.landscape.epsilon,
        results_dir=Path(cfg.project.results_dir),
    )

    print("\nSharpness summary :")
    for d in sharpness:
        print(f"  {d['label']:20s} | sharpness={d['sharpness']:.5f} | F1={d['val_f1']:.4f}")


def main() -> None:
    """Point d'entrée CLI pour l'analyse du loss landscape."""
    typer.run(_cli_landscape)


if __name__ == "__main__":
    main()
