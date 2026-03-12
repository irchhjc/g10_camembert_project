# ============================================================
# G10 — CamemBERT Allociné : Configuration centrale (P02)
# ============================================================

from types import SimpleNamespace


def _ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


CFG = _ns(
    project=_ns(
        name="g10-camembert-allocine",
        group="G10",
        protocol="P02",
        seed=42,
        results_dir="results",
        figures_dir="results/figures",
        models_dir="results/models",
    ),

    # ── Dataset ────────────────────────────────────────────────
    dataset=_ns(
        name="allocine",
        text_col="review",
        label_col="label",
        label_names=["Négatif", "Positif"],
        max_seq_len=256,
        # Tailles sous-ensembles (adaptation matérielle)
        n_train_per_class=500,      # 1000 total
        n_val_per_class=150,        # 300 total
        n_test_per_class=150,       # 300 total
        # Tailles réduites pour les trials Optuna
        n_optuna_train_per_class=250,
        n_optuna_val_per_class=100,
    ),

    # ── Modèle ─────────────────────────────────────────────────
    model=_ns(
        name="camembert-base",
        num_labels=2,
        max_seq_len=256,
    ),

    # ── Entraînement ────────────────────────────────────────────
    training=_ns(
        batch_size=16,
        grad_accum_steps=2,         # batch effectif = 32
        num_epochs=3,
        warmup_ratio=0.1,
        early_stopping_patience=2,
        lr_baseline=2.0e-5,
        weight_decay_baseline=1.0e-4,
        dropout_baseline=0.1,
    ),

    # ── Entraînement final (post-Optuna) ─────────────────────────
    final_training=_ns(
        num_epochs=5,
        early_stopping_patience=3,
    ),

    # ── Protocole P02 — Grid Search ─────────────────────────────
    protocol_p02=_ns(
        weight_decay_grid=[1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        dropout_grid=[0.0, 0.1, 0.3],
        grid_num_epochs=2,          # Réduit pour isolation des effets
        grid_lr=2.0e-5,             # LR fixé pour isoler wd × dp
    ),

    # ── Optimisation Bayésienne (Optuna) ─────────────────────────
    optuna=_ns(
        study_name="g10_p02_regularisation",
        n_trials=20,
        direction="maximize",
        sampler="TPE",
        n_startup_trials=3,         # Trials random avant TPE
        pruner="MedianPruner",
        lr_min=1.0e-6,
        lr_max=5.0e-4,
        db_path="results/optuna.db",
    ),

    # ── Analyse Loss Landscape ───────────────────────────────────
    landscape=_ns(
        n_points=8,                 # Points d'évaluation sur [-ε, +ε]
        epsilon=0.05,               # Amplitude de perturbation
        n_samples=50,               # Batches pour estimation loss
        dropout_values_to_compare=[0.0, 0.1, 0.3],
    ),

    # ── Logging ─────────────────────────────────────────────────
    logging=_ns(
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        file="results/experiment.log",
    ),
)
