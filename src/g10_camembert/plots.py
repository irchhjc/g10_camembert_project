"""
Module de visualisation — G10 Protocole P02.

Génère toutes les figures utilisées dans le rapport :
1. Distribution des longueurs (EDA)
2. Courbes baseline (F1, Loss)
3. Heatmaps Grid P02 (gap + F1-val)
4. Courbes de convergence comparatives
5. Visualisations Optuna (historique, distributions)
6. Loss landscape 1D + scatter sharpness
7. Courbes d'entraînement final
8. Figure de synthèse multi-panneaux
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style global ──────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("husl", 8)
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_theme(style="whitegrid")


def save_or_show(fig: plt.Figure, path: Path | None = None) -> None:
    """Sauvegarde et/ou affiche la figure."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


# ── 1. Distribution des longueurs ─────────────────────────────────────────────

def plot_length_distribution(
    lengths_neg: list[int],
    lengths_pos: list[int],
    trunc_at: int = 256,
    save_path: Path | None = None,
) -> None:
    """
    Histogramme et CDF de la distribution des longueurs par classe.

    Args:
        lengths_neg: Longueurs (en mots) des critiques négatives.
        lengths_pos: Longueurs (en mots) des critiques positives.
        trunc_at:    Seuil de troncature (affiché en rouge).
        save_path:   Chemin de sauvegarde.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    all_lengths = lengths_neg + lengths_pos

    # Histogramme par classe
    axes[0].hist(lengths_neg, bins=50, alpha=0.7, color=PALETTE[0],
                 label="Négatif", edgecolor="white")
    axes[0].hist(lengths_pos, bins=50, alpha=0.7, color=PALETTE[3],
                 label="Positif", edgecolor="white")
    axes[0].axvline(trunc_at, color="red", lw=2, ls="--", label=f"Troncature ({trunc_at})")
    axes[0].set_xlabel("Longueur (mots)", fontsize=11)
    axes[0].set_ylabel("Fréquence", fontsize=11)
    axes[0].set_title("Distribution des longueurs par classe", fontsize=12)
    axes[0].legend()

    # Distribution cumulée
    sorted_l = np.sort(all_lengths)
    cum = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
    covered = sum(1 for l in all_lengths if l <= trunc_at) / len(all_lengths) * 100
    axes[1].plot(sorted_l, cum * 100, color=PALETTE[2], lw=2.5)
    axes[1].axvline(trunc_at, color="red", lw=2, ls="--",
                    label=f"{trunc_at} mots ({covered:.0f}% couvert)")
    axes[1].set_xlabel("Longueur (mots)", fontsize=11)
    axes[1].set_ylabel("% critiques couvertes", fontsize=11)
    axes[1].set_title("Distribution cumulée des longueurs", fontsize=12)
    axes[1].legend()

    fig.suptitle("Analyse des longueurs — Dataset Allociné (D05)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


# ── 2. Courbes d'entraînement ─────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    title: str = "Courbes d'entraînement",
    save_path: Path | None = None,
) -> None:
    """
    Courbes F1-score et loss (train/val) par époque.

    Args:
        history:   Dict {train_f1, val_f1, train_loss, val_loss}.
        title:     Titre du graphique.
        save_path: Chemin de sauvegarde.
    """
    epochs = range(1, len(history["train_f1"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_f1"], "o-", color=PALETTE[0], lw=2.5, label="Train")
    axes[0].plot(epochs, history["val_f1"],   "s--", color=PALETTE[2], lw=2.5, label="Validation")
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("F1-score macro")
    axes[0].set_title("F1-score macro"); axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], "o-",  color=PALETTE[0], lw=2.5, label="Train")
    axes[1].plot(epochs, history["val_loss"],   "s--", color=PALETTE[2], lw=2.5, label="Validation")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Loss"); axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


# ── 3. Heatmaps Grid P02 ──────────────────────────────────────────────────────

def plot_heatmaps_p02(
    df_grid: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Heatmaps du gap de généralisation et du F1-val pour le Grid Search P02.

    Args:
        df_grid:   DataFrame avec colonnes [weight_decay, dropout, gap, val_f1].
        save_path: Chemin de sauvegarde.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wd_labels = [f"{w:.0e}" for w in sorted(df_grid["weight_decay"].unique())]
    dp_labels = [f"{d:.1f}" for d in sorted(df_grid["dropout"].unique())]

    pivot_gap = df_grid.pivot_table(index="weight_decay", columns="dropout", values="gap")
    sns.heatmap(pivot_gap, ax=axes[0], cmap="Reds", annot=True, fmt=".3f",
                linewidths=0.8, cbar_kws={"label": "Gap"},
                xticklabels=dp_labels, yticklabels=wd_labels)
    axes[0].set_title("Gap (F1_train − F1_val)\n← plus faible = mieux", fontsize=11)
    axes[0].set_xlabel("Dropout"); axes[0].set_ylabel("Weight Decay")

    pivot_f1 = df_grid.pivot_table(index="weight_decay", columns="dropout", values="val_f1")
    sns.heatmap(pivot_f1, ax=axes[1], cmap="Greens", annot=True, fmt=".3f",
                linewidths=0.8, cbar_kws={"label": "F1 macro"},
                xticklabels=dp_labels, yticklabels=wd_labels)
    axes[1].set_title("F1-score macro (validation)\n← plus élevé = mieux", fontsize=11)
    axes[1].set_xlabel("Dropout"); axes[1].set_ylabel("")

    fig.suptitle("Protocole P02 — Grid Search : Régularisation × Généralisation\n"
                 "CamemBERT-base | Allociné | G10", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


# ── 4. Optuna convergence ─────────────────────────────────────────────────────

def plot_optuna_results(
    study: Any,
    save_path: Path | None = None,
) -> None:
    """
    Visualisations de l'étude Optuna : convergence + distributions.

    Args:
        study:     Objet Study Optuna.
        save_path: Chemin de sauvegarde.
    """
    import optuna
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # (a) Historique de convergence
    trial_nums = [t.number for t in completed]
    f1_vals = [t.value for t in completed]
    best_so_far = np.maximum.accumulate(f1_vals)
    axes[0].scatter(trial_nums, f1_vals, alpha=0.6, color=PALETTE[1], s=50, zorder=3)
    axes[0].plot(trial_nums, best_so_far, color=PALETTE[5], lw=2.5, label="Meilleur cumulé")
    axes[0].set_xlabel("Trial #"); axes[0].set_ylabel("F1-score macro (val)")
    axes[0].set_title("Convergence Optuna (TPE)"); axes[0].legend()

    # (b) Distribution par dropout
    for i, dp_val in enumerate(sorted(set(t.params.get("dropout") for t in completed))):
        vals = [t.value for t in completed if t.params.get("dropout") == dp_val]
        axes[1].scatter([dp_val] * len(vals), vals, alpha=0.7, s=60,
                        color=PALETTE[i * 2], label=f"dropout={dp_val}", zorder=3)
    axes[1].set_xlabel("Dropout"); axes[1].set_ylabel("F1-score macro (val)")
    axes[1].set_title("Distribution F1 par dropout"); axes[1].legend(fontsize=9)

    # (c) Distribution par weight_decay
    for i, wd_val in enumerate(sorted(set(t.params.get("weight_decay") for t in completed))):
        vals = [t.value for t in completed if t.params.get("weight_decay") == wd_val]
        axes[2].scatter([np.log10(wd_val)] * len(vals), vals, alpha=0.7, s=60,
                        color=PALETTE[i * 2], label=f"wd={wd_val:.0e}", zorder=3)
    axes[2].set_xlabel("log10(weight_decay)"); axes[2].set_ylabel("F1-score macro (val)")
    axes[2].set_title("Distribution F1 par weight_decay"); axes[2].legend(fontsize=9)

    fig.suptitle("Optimisation Bayésienne Optuna/TPE — 20 trials", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


# ── 5. Loss Landscape ─────────────────────────────────────────────────────────

def plot_loss_landscape(
    landscape_results: dict[str, tuple],
    sharpness_data: list[dict],
    save_path: Path | None = None,
) -> None:
    """
    Loss landscape 1D + scatter sharpness vs F1-val.

    Args:
        landscape_results: Dict label → (alphas, losses).
        sharpness_data:    Liste de dicts {label, sharpness, val_f1, dropout}.
        save_path:         Chemin de sauvegarde.
    """
    styles = [
        {"color": PALETTE[0], "ls": "-",  "marker": "o"},
        {"color": PALETTE[3], "ls": "--", "marker": "s"},
        {"color": PALETTE[5], "ls": ":",  "marker": "^"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (lbl, (alphas, losses)) in enumerate(landscape_results.items()):
        s = styles[i % len(styles)]
        from g10_camembert.metrics import compute_sharpness
        base = losses[len(losses) // 2]
        sharp = compute_sharpness(base, losses)
        axes[0].plot(alphas, losses, color=s["color"], lw=2.5, ls=s["ls"],
                     marker=s["marker"], ms=6, markerfacecolor="white",
                     label=f"{lbl}  (sharpness={sharp:.4f})")

    axes[0].axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5, label="θ* (convergence)")
    axes[0].set_xlabel("Direction de perturbation (α)", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Loss landscape 1D\n(perturbation filtre-normalisée)", fontsize=12)
    axes[0].legend(fontsize=9)

    # Scatter sharpness vs F1
    for i, d in enumerate(sharpness_data):
        axes[1].scatter(d["sharpness"], d["val_f1"], color=styles[i % len(styles)]["color"],
                        s=150, zorder=5, edgecolors="white", linewidths=1.5)
        axes[1].annotate(d["label"], (d["sharpness"], d["val_f1"]),
                         xytext=(6, 4), textcoords="offset points", fontsize=10)

    axes[1].set_xlabel("Sharpness", fontsize=11)
    axes[1].set_ylabel("F1-val", fontsize=11)
    axes[1].set_title("Sharpness vs Généralisation", fontsize=12)

    fig.suptitle("Analyse du Loss Landscape — CamemBERT-base", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


def main() -> None:
    """Point d'entrée CLI pour la génération des figures."""
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        results_dir: str = typer.Option("results", help="Dossier résultats"),
    ) -> None:
        print("Génération des figures depuis les résultats sauvegardés...")
        # Les figures sont générées lors de l'exécution du pipeline complet

    app()


if __name__ == "__main__":
    main()
