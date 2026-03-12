#!/usr/bin/env python
"""
Analyse du Loss Landscape — Protocole P02.
Usage : poetry run python run_landscape.py
"""
import json
from pathlib import Path

from configs.config import CFG
from g10_camembert.loader import load_allocine, prepare_splits
from g10_camembert.dataset import AllocinéDataset
from g10_camembert.camembert import load_camembert, load_tokenizer, get_device
from g10_camembert.trainer import train_model
from g10_camembert.loss_landscape import analyze_landscape_multiple_configs

if __name__ == "__main__":
    device = get_device()
    dataset = load_allocine()
    train_s, val_s, _ = prepare_splits(
        dataset,
        n_train=CFG.dataset.n_train_per_class,
        n_val=CFG.dataset.n_val_per_class,
        seed=CFG.project.seed,
    )
    tokenizer = load_tokenizer(CFG.model.name)
    train_ds = AllocinéDataset(train_s, tokenizer, CFG.dataset.max_seq_len)
    val_ds   = AllocinéDataset(val_s,   tokenizer, CFG.dataset.max_seq_len)

    best_params_path = Path(CFG.project.results_dir) / "best_params.json"
    best_wd = CFG.training.weight_decay_baseline
    best_lr = CFG.training.lr_baseline
    if best_params_path.exists():
        with open(best_params_path) as f:
            bp = json.load(f)
        best_wd = bp["best_params"].get("weight_decay", best_wd)
        best_lr = bp["best_params"].get("learning_rate", best_lr)

    models_and_labels = []
    for dp in CFG.landscape.dropout_values_to_compare:
        model = load_camembert(dropout=dp, device=device)
        result = train_model(
            model, train_ds, val_ds,
            lr=best_lr, weight_decay=best_wd,
            batch_size=CFG.training.batch_size,
            num_epochs=3, verbose=False,
        )
        models_and_labels.append((model, f"dropout={dp:.1f}", result.best_val_f1, dp))

    _, sharpness = analyze_landscape_multiple_configs(
        models_and_labels, val_ds, device=device,
        n_points=CFG.landscape.n_points,
        epsilon=CFG.landscape.epsilon,
        results_dir=Path(CFG.project.results_dir),
    )
    print("\nSharpness summary :")
    for d in sharpness:
        print(f"  {d['label']:20s} | sharpness={d['sharpness']:.5f} | F1={d['val_f1']:.4f}")
