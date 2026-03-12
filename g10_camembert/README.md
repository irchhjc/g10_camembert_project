# 🎬 G10 — CamemBERT Fine-tuning sur Allociné
## Protocole P02 : Régularisation & Généralisation

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Question de recherche

> **Comment le `weight_decay` et le `dropout` affectent-ils la généralisation de
> CamemBERT fine-tuné sur des critiques de films en français ?**

## 📂 Structure du projet

```
g10_camembert/
├── pyproject.toml              # Configuration Poetry & dépendances
├── README.md
├── configs/
│   └── config.py               # Configuration centralisée (Python pur)
├── run_baseline.py             # Exécution entraînement baseline
├── run_grid_search.py          # Exécution Grid Search P02
├── run_optuna.py               # Exécution optimisation Optuna
├── run_landscape.py            # Exécution analyse Loss Landscape
├── run_pipeline.py             # Pipeline complet (toutes étapes)
├── src/
│   └── g10_camembert/          # Package Python (fichiers plats, sans sous-dossiers)
│       ├── __init__.py
│       ├── dataset.py          # AllocinéDataset (PyTorch)
│       ├── loader.py           # Chargement & sous-échantillonnage
│       ├── camembert.py        # Chargement CamemBERT configurable
│       ├── trainer.py          # Boucle AdamW + warmup + early stopping
│       ├── grid_search.py      # Grid Search P02 (wd × dropout)
│       ├── optuna_search.py    # Optimisation Bayésienne Optuna/TPE
│       ├── loss_landscape.py   # Paysage de perte & sharpness
│       ├── plots.py            # Toutes les visualisations
│       ├── metrics.py          # F1-score, gap, sharpness
│       ├── seed.py             # Reproductibilité
│       └── config_loader.py    # Chargement de la configuration
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_trainer.py
├── notebooks/
│   └── g10-projet-mloptimisation-enrichi.ipynb
└── results/
    ├── figures/                # Graphiques générés
    └── models/                 # Checkpoints sauvegardés
```

## 🚀 Installation

### Prérequis
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- GPU CUDA recommandé (CPU supporté avec adaptation)

### Installation rapide

```bash
# Cloner le projet
git clone <repo-url>
cd g10_camembert

# Installer les dépendances avec Poetry
poetry install

# (Optionnel) Activer l'environnement virtuel
poetry shell
```

### Installation avec les extras de visualisation

```bash
poetry install --with viz
```

## ⚙️ Configuration

Toute la configuration est centralisée dans `configs/config.py` (Python pur, pas de YAML) :

```python
CFG = _ns(
    model=_ns(
        name="camembert-base",
        max_seq_len=256,
        num_labels=2,
    ),
    training=_ns(
        batch_size=16,
        grad_accum_steps=2,
        num_epochs=3,
        warmup_ratio=0.1,
        early_stopping_patience=2,
        lr_baseline=2.0e-5,
        weight_decay_baseline=1.0e-4,
        dropout_baseline=0.1,
    ),
    protocol_p02=_ns(
        weight_decay_grid=[1e-5, 1e-4, 1e-3, 1e-2],
        dropout_grid=[0.0, 0.1, 0.3],
        grid_num_epochs=2,
        grid_lr=2.0e-5,
    ),
    ...
)
```

Modifier directement `configs/config.py` pour ajuster les hyperparamètres.

## 🏃 Usage

### Scripts d'exécution (racine du projet)

```bash
# Entraînement baseline
poetry run python run_baseline.py

# Grid Search P02
poetry run python run_grid_search.py

# Optimisation Bayésienne Optuna
poetry run python run_optuna.py         # 20 trials par défaut
poetry run python run_optuna.py 30      # nb trials custom

# Analyse du Loss Landscape
poetry run python run_landscape.py

# Pipeline complet (toutes les étapes)
poetry run python run_pipeline.py
```

### Via les entry points Poetry (CLI)

```bash
pretry run g10-train    --config configs/config.py
poetry run g10-optimize --method grid   --config configs/config.py
poetry run g10-optimize --method optuna --n-trials 20 --config configs/config.py
poetry run g10-landscape --config configs/config.py
```

## 📊 Résultats clés

| Configuration | F1-val | F1-test | Gap |
|--------------|--------|---------|-----|
| Baseline (wd=1e-4, dp=0.1, lr=2e-5) | 0.9067 | 0.9367 | 0.043 |
| Grid optimal (wd=1e-3, dp=0.0) | 0.9067 | — | 0.025 |
| **Optuna optimal** (wd=1e-3, dp=0.1, lr=5e-5) | **0.9333** | **0.9600** | 0.060 |

**Gain vs baseline : +2,5% (+40% réduction taux d'erreur)**

## 🧪 Tests

```bash
# Lancer tous les tests
poetry run pytest

# Avec rapport de couverture
poetry run pytest --cov=src/g10_camembert --cov-report=html
```

## 📝 Notebook enrichi

Le notebook principal avec interprétations complètes est disponible dans
`notebooks/g10-projet-mloptimisation-enrichi.ipynb`.

```bash
poetry run jupyter notebook notebooks/
```

## 🔬 Protocole P02 — Détails

L'expérience explore l'espace :
- **weight_decay** ∈ {1e-5, 1e-4, 1e-3, 1e-2} (4 valeurs)
- **dropout** ∈ {0.0, 0.1, 0.3} (3 valeurs)
- **learning_rate** ∈ [1e-6, 5e-4] log-scale (continu, Optuna)

**Conclusions principales :**
1. dropout=0.1 est optimal pour CamemBERT — valeur standard RoBERTa
2. dropout=0.3 est destructeur (effondrement performances à F1≈0.47)
3. Le weight decay a un effet marginal dans ce régime
4. L'optimisation bayésienne est décisive via l'exploration du learning rate

## 📚 Références

- Martin et al. (2020) — CamemBERT: a Tasty French Language Model
- Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization (TPE)
- Loshchilov & Hutter (2019) — Decoupled Weight Decay Regularization (AdamW)
- Li et al. (2018) — Visualizing the Loss Landscape of Neural Nets
- Keskar et al. (2017) — On Large-Batch Training for Deep Learning

## 📄 Licence

MIT — Groupe G10, 2026
