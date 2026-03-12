#!/usr/bin/env python
"""
Entraînement baseline CamemBERT — Allociné.
Usage : poetry run python run_baseline.py
"""
from configs.config import CFG
from g10_camembert.loader import load_allocine, prepare_splits
from g10_camembert.dataset import AllocinéDataset
from g10_camembert.camembert import load_camembert, load_tokenizer, get_device
from g10_camembert.trainer import train_model, evaluate
from loguru import logger

if __name__ == "__main__":
    device = get_device()

    dataset = load_allocine()
    train_s, val_s, test_s = prepare_splits(
        dataset,
        n_train=CFG.dataset.n_train_per_class,
        n_val=CFG.dataset.n_val_per_class,
        n_test=CFG.dataset.n_test_per_class,
        seed=CFG.project.seed,
    )
    tokenizer = load_tokenizer(CFG.model.name, CFG.model.max_seq_len)
    train_ds = AllocinéDataset(train_s, tokenizer, CFG.dataset.max_seq_len)
    val_ds   = AllocinéDataset(val_s,   tokenizer, CFG.dataset.max_seq_len)
    test_ds  = AllocinéDataset(test_s,  tokenizer, CFG.dataset.max_seq_len)

    model = load_camembert(dropout=CFG.training.dropout_baseline, device=device)
    result = train_model(
        model, train_ds, val_ds,
        lr=CFG.training.lr_baseline,
        weight_decay=CFG.training.weight_decay_baseline,
        batch_size=CFG.training.batch_size,
        grad_accum=CFG.training.grad_accum_steps,
        num_epochs=CFG.training.num_epochs,
        device=device,
        seed=CFG.project.seed,
    )
    test_metrics = evaluate(model, test_ds, device=device, verbose=True)
    logger.info(f"F1-test baseline : {test_metrics['f1_macro']:.4f}")
