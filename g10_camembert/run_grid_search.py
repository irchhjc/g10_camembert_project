#!/usr/bin/env python
"""
Grid Search P02 : exploration weight_decay × dropout (12 configurations).
Usage : poetry run python run_grid_search.py
"""
from pathlib import Path

from configs.config import CFG
from g10_camembert.loader import load_allocine, prepare_splits
from g10_camembert.dataset import AllocinéDataset
from g10_camembert.camembert import load_tokenizer
from g10_camembert.grid_search import run_grid_search

if __name__ == "__main__":
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

    df, _ = run_grid_search(
        train_ds, val_ds,
        weight_decay_grid=CFG.protocol_p02.weight_decay_grid,
        dropout_grid=CFG.protocol_p02.dropout_grid,
        lr=CFG.protocol_p02.grid_lr,
        num_epochs=CFG.protocol_p02.grid_num_epochs,
        results_dir=Path(CFG.project.results_dir),
        seed=CFG.project.seed,
    )
    print(df.to_string(index=False))
