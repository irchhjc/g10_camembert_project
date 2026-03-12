"""
Boucle d'entraînement CamemBERT — AdamW + Scheduler linéaire + Early Stopping.

Implémentations clés :
- AdamW avec exclusion sélective du weight decay (Loshchilov & Hutter, 2019)
- Gradient accumulation pour simulation de grands batches sur CPU
- Scheduler linéaire avec warmup
- Early stopping sur F1-val (métrique principale P02)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import typer

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from g10_camembert.utils.metrics import compute_metrics
from g10_camembert.utils.seed import set_seed


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    """Historique complet d'un entraînement."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Résultat complet d'un entraînement."""
    best_val_f1: float
    history: TrainingHistory
    time_s: float
    best_epoch: int


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def get_optimizer_params(
    model: Any,
    weight_decay: float,
) -> list[dict]:
    """
    Construit les groupes de paramètres pour AdamW.

    Exclut les biais et paramètres LayerNorm du weight decay,
    conformément à la pratique standard pour les Transformers
    (Loshchilov & Hutter, 2019 ; Devlin et al., 2019).

    Args:
        model:        Modèle CamemBERT.
        weight_decay: Coefficient de pénalisation L2.

    Returns:
        Liste de groupes de paramètres pour AdamW.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


@torch.no_grad()
def evaluate(
    model: Any,
    dataset: Dataset,
    batch_size: int = 32,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Évalue le modèle sur un dataset.

    Args:
        model:      Modèle CamemBERT.
        dataset:    Dataset PyTorch.
        batch_size: Taille de batch pour l'évaluation.
        device:     Device cible.
        verbose:    Afficher le rapport complet.

    Returns:
        Dictionnaire {f1_macro, accuracy, loss}.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss, all_labels, all_preds = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)

    if verbose:
        from g10_camembert.utils.metrics import full_classification_report
        label_names = ["Négatif", "Positif"]
        print(full_classification_report(all_labels, all_preds, label_names))

    return metrics


# ── Boucle principale ─────────────────────────────────────────────────────────

def train_model(
    model: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    grad_accum: int = 2,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    device: torch.device | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainingResult:
    """
    Entraîne CamemBERT avec le protocole P02.

    Protocole d'entraînement :
    1. AdamW avec exclusion biais/LayerNorm du weight decay
    2. Scheduler linéaire avec warmup (10% des steps)
    3. Gradient accumulation (batch effectif = batch_size × grad_accum)
    4. Early stopping sur F1-val (patience configurable)

    Args:
        model:                   Modèle CamemBERT pré-chargé.
        train_dataset:           Dataset d'entraînement.
        val_dataset:             Dataset de validation.
        lr:                      Taux d'apprentissage initial.
        weight_decay:            Coefficient de régularisation L2 (P02).
        batch_size:              Taille de mini-batch.
        grad_accum:              Pas d'accumulation de gradient.
        num_epochs:              Nombre maximal d'époques.
        warmup_ratio:            Fraction des steps pour le warmup.
        early_stopping_patience: Patience pour l'early stopping.
        device:                  Device cible (auto si None).
        seed:                    Graine aléatoire.
        verbose:                 Afficher les métriques par époque.

    Returns:
        TrainingResult avec historique complet.
    """
    set_seed(seed)

    if device is None:
        device = next(model.parameters()).device

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )

    # Groupes de paramètres AdamW (exclusion biais/LayerNorm)
    opt_params = get_optimizer_params(model, weight_decay)
    optimizer = AdamW(opt_params, lr=lr)

    # Scheduler linéaire avec warmup
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = TrainingHistory()
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── Phase entraînement ──
        model.train()
        train_loss, all_labels, all_preds = 0.0, [], []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()
            train_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds)

            # Mise à jour des poids tous les grad_accum steps
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ── Phase validation ──
        val_metrics = evaluate(model, val_dataset, device=device)
        train_metrics = compute_metrics(all_labels, all_preds)

        t_loss = train_loss / len(train_loader)
        gap = train_metrics["f1_macro"] - val_metrics["f1_macro"]

        # Enregistrement dans l'historique
        history.train_loss.append(t_loss)
        history.val_loss.append(val_metrics["loss"])
        history.train_f1.append(train_metrics["f1_macro"])
        history.val_f1.append(val_metrics["f1_macro"])
        history.train_acc.append(train_metrics["accuracy"])
        history.val_acc.append(val_metrics["accuracy"])

        if verbose:
            logger.info(
                f"  Époque {epoch}/{num_epochs} — "
                f"loss={t_loss:.4f}/{val_metrics['loss']:.4f} | "
                f"F1={train_metrics['f1_macro']:.4f}/{val_metrics['f1_macro']:.4f} | "
                f"gap={gap:+.4f}"
            )

        # Early stopping
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"  Early stopping déclenché à l'époque {epoch}")
                break

    elapsed = time.time() - t_start
    if verbose:
        logger.info(
            f"  ✅ Terminé en {elapsed:.0f}s | "
            f"Meilleur F1-val : {best_val_f1:.4f} (époque {best_epoch})"
        )

    return TrainingResult(
        best_val_f1=best_val_f1,
        history=history,
        time_s=elapsed,
        best_epoch=best_epoch,
    )


# ── CLI Entry point ────────────────────────────────────────────────────────────

def _cli_train(
    config: str = typer.Option("configs/config.py", help="Chemin config Python"),
) -> None:
    from g10_camembert.utils.config import load_config
    from g10_camembert.data import load_allocine, prepare_splits, AllocinéDataset
    from g10_camembert.models.camembert import load_camembert, load_tokenizer, get_device

    cfg = load_config(config)
    device = get_device()

    dataset = load_allocine()
    train_s, val_s, test_s = prepare_splits(
        dataset,
        n_train=cfg.dataset.n_train_per_class,
        n_val=cfg.dataset.n_val_per_class,
        n_test=cfg.dataset.n_test_per_class,
        seed=cfg.project.seed,
    )
    tokenizer = load_tokenizer(cfg.model.name, cfg.model.max_seq_len)
    train_ds = AllocinéDataset(train_s, tokenizer, cfg.dataset.max_seq_len)
    val_ds = AllocinéDataset(val_s, tokenizer, cfg.dataset.max_seq_len)
    test_ds = AllocinéDataset(test_s, tokenizer, cfg.dataset.max_seq_len)

    model = load_camembert(
        dropout=cfg.training.dropout_baseline,
        device=device,
    )
    result = train_model(
        model, train_ds, val_ds,
        lr=cfg.training.lr_baseline,
        weight_decay=cfg.training.weight_decay_baseline,
        batch_size=cfg.training.batch_size,
        grad_accum=cfg.training.grad_accum_steps,
        num_epochs=cfg.training.num_epochs,
        device=device,
        seed=cfg.project.seed,
    )
    test_metrics = evaluate(model, test_ds, device=device, verbose=True)
    logger.info(f"F1-test baseline : {test_metrics['f1_macro']:.4f}")


def main() -> None:
    """Point d'entrée CLI pour l'entraînement baseline."""
    typer.run(_cli_train)


if __name__ == "__main__":
    main()
