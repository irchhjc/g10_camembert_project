"""
Chargement et configuration de CamemBERT-base pour la classification.

Le dropout est appliqué uniformément aux trois niveaux de l'architecture :
- hidden_dropout_prob       : couches cachées du Transformer
- attention_probs_dropout_prob : têtes d'attention multi-head
- classifier_dropout        : tête de classification linéaire

Cette uniformité est requise par le protocole P02 pour isoler l'effet
du paramètre dropout de manière cohérente.
"""

from __future__ import annotations

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    PreTrainedTokenizerBase,
)


MODEL_NAME = "camembert-base"
NUM_LABELS = 2


def load_tokenizer(
    model_name: str = MODEL_NAME,
    max_length: int = 512,
) -> PreTrainedTokenizerBase:
    """
    Charge le tokenizer SentencePiece BPE de CamemBERT.

    Args:
        model_name: Identifiant HuggingFace du modèle.
        max_length: Longueur maximale pour le tokenizer.

    Returns:
        Tokenizer CamemBERT initialisé.
    """
    logger.info(f"Chargement du tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
    )
    logger.info(f"Vocabulaire : {tokenizer.vocab_size:,} tokens (SentencePiece BPE)")
    return tokenizer


def load_camembert(
    dropout: float = 0.1,
    num_labels: int = NUM_LABELS,
    model_name: str = MODEL_NAME,
    device: torch.device | None = None,
) -> CamembertForSequenceClassification:
    """
    Charge CamemBERT-base avec dropout configurable (protocole P02).

    Le dropout est appliqué de manière identique aux trois composantes :
    - hidden_dropout_prob = dropout
    - attention_probs_dropout_prob = dropout
    - classifier_dropout = dropout

    Cette uniformité garantit que les comparaisons entre configurations
    mesurent bien l'effet global du dropout sur l'architecture.

    Args:
        dropout:    Probabilité de dropout (0.0 à 0.5 recommandé).
        num_labels: Nombre de classes (2 pour binaire).
        model_name: Identifiant HuggingFace.
        device:     Device cible (auto-détecté si None).

    Returns:
        Modèle CamemBERT configuré et déplacé sur le device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration avec dropout uniforme
    config = CamembertConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        classifier_dropout=dropout,
    )

    model = CamembertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # Tête de classification nouvellement initialisée
    )

    model = model.to(device)

    # Résumé des paramètres
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"CamemBERT chargé | dropout={dropout} | "
        f"{total/1e6:.1f}M params ({trainable/1e6:.1f}M entraînables) | "
        f"device={device}"
    )

    return model


def get_device() -> torch.device:
    """Auto-détecte le device optimal (CUDA > CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
        logger.warning("GPU non disponible — utilisation CPU (entraînement lent)")
    else:
        logger.info(f"GPU disponible : {torch.cuda.get_device_name(0)}")
    return device
