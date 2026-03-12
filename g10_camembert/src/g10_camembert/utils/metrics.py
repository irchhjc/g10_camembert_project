"""
Métriques d'évaluation pour le protocole P02.

Métriques implémentées :
- F1-score macro
- Accuracy
- Gap de généralisation (train/val)
- Sharpness (loss landscape)
"""

from typing import Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    labels: list[int],
    preds: list[int],
) -> dict[str, float]:
    """
    Calcule accuracy et F1-score macro.

    Args:
        labels: Étiquettes réelles.
        preds:  Prédictions du modèle.

    Returns:
        Dictionnaire {accuracy, f1_macro}.
    """
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    return {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "f1_macro": float(
            f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        ),
    }


def generalization_gap(
    train_f1: float,
    val_f1: float,
) -> dict[str, float]:
    """
    Calcule le gap de généralisation (P02).

    Le gap mesure l'écart entre les performances d'entraînement et de validation.
    Un gap élevé indique un sur-apprentissage.

    Args:
        train_f1: F1-score sur le jeu d'entraînement.
        val_f1:   F1-score sur le jeu de validation.

    Returns:
        Dictionnaire {train_f1, val_f1, gap, gap_pct}.
    """
    gap = train_f1 - val_f1
    gap_pct = (gap / train_f1 * 100) if train_f1 > 0 else 0.0
    return {
        "train_f1": train_f1,
        "val_f1": val_f1,
        "gap": gap,
        "gap_pct": gap_pct,
    }


def compute_sharpness(
    base_loss: float,
    perturbed_losses: list[float],
) -> float:
    """
    Calcule la sharpness du minimum convergé.

    Formule : Sharpness(θ*) = (1/N) Σ |L(θ* + εdᵢ) - L(θ*)|

    Source : Li et al. (2018) — Visualizing the Loss Landscape of Neural Nets.

    Args:
        base_loss:        Loss au point de convergence θ*.
        perturbed_losses: Losses après perturbation dans différentes directions.

    Returns:
        Sharpness scalar (float).
    """
    return float(np.mean([abs(loss - base_loss) for loss in perturbed_losses]))


def full_classification_report(
    labels: list[int],
    preds: list[int],
    label_names: list[str] | None = None,
) -> str:
    """
    Génère un rapport de classification complet.

    Args:
        labels:      Étiquettes réelles.
        preds:       Prédictions.
        label_names: Noms des classes (optionnel).

    Returns:
        Rapport formaté sous forme de chaîne.
    """
    target_names = label_names or [str(i) for i in sorted(set(labels))]
    return classification_report(
        labels, preds, target_names=target_names, zero_division=0
    )


def confusion_matrix_data(
    labels: list[int],
    preds: list[int],
) -> np.ndarray:
    """
    Calcule la matrice de confusion.

    Args:
        labels: Étiquettes réelles.
        preds:  Prédictions.

    Returns:
        Matrice de confusion numpy.
    """
    return confusion_matrix(labels, preds)
