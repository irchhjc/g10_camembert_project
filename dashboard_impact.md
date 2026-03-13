# Dashboard d'impact — Résultats CamemBERT G10

Ce tableau de bord synthétise les résultats clés des expériences de fine-tuning CamemBERT sur Allociné.

## 🏆 Comparatif des meilleures configurations

| Configuration | F1 (val) | F1 (test) | Gap (train-val) |
|---------------|----------|-----------|-----------------|
| Baseline (wd=1e-4, dp=0.1) | 0.91 | 0.94 | 0.04 |
| Grid Optimal (wd=1e-2, dp=?) | 0.91 | — | 0.03 |
| Optuna Optimal (wd=1e-3, dp=0.1, lr=5e-5) | **0.93** | **0.96** | 0.06 |

- **Gain Optuna vs baseline :** +2,5 points F1 (+40% réduction du taux d'erreur)

## 📈 Impact du Dropout sur la Sharpness

| Dropout | Sharpness | F1 (val) |
|---------|----------|----------|
| 0.0     | 0.11     | 0.93     |
| 0.1     | 0.30     | 0.92     |
| 0.3     | 0.11     | 0.92     |

- **dropout=0.1** (valeur RoBERTa) : bon compromis généralisation/stabilité
- **dropout=0.3** : effondrement des performances

## 🔍 Meilleurs hyperparamètres trouvés (Optuna)

- **weight_decay** : 0.001
- **dropout** : 0.1
- **learning_rate** : 5e-5

## 🧩 Analyse Grid Search (extrait)

| weight_decay | dropout | F1 (val) | Gap |
|--------------|---------|----------|-----|
| 1e-5         | 0.0     | 0.92     | 0.01|
| 0.0001       | 0.1     | 0.89     | 0.02|
| 0.01         | 0.3     | 0.62     | -0.03|

## 📊 Visualisation interactive

Pour explorer les essais et l'optimisation :

```bash
poetry run optuna-dashboard sqlite:///results/optuna_final.db
```

Ouvrez le lien affiché dans votre navigateur pour naviguer dans les résultats détaillés.

---

*Ce dashboard a été généré automatiquement à partir des fichiers de résultats du projet.*
