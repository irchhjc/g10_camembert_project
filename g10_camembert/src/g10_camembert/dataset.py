"""
Dataset PyTorch pour CamemBERT — Allociné.

Implémente la tokenisation SentencePiece BPE en batch, avec padding
et troncature à MAX_SEQ_LEN. La tokenisation est pré-calculée en mémoire
pour éviter les goulots d'étranglement pendant l'entraînement.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class AllocinéDataset(Dataset):
    """
    Dataset PyTorch pour l'analyse de sentiment Allociné.

    La tokenisation est effectuée une seule fois à l'initialisation
    (in-memory tokenization) pour maximiser la vitesse de chargement
    pendant l'entraînement.

    Args:
        samples:     Liste de dicts {review: str, label: int}.
        tokenizer:   Tokenizer CamemBERT SentencePiece.
        max_length:  Longueur maximale de séquence (défaut : 256).
        text_col:    Nom de la colonne texte (défaut : 'review').
        label_col:   Nom de la colonne étiquette (défaut : 'label').
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        text_col: str = "review",
        label_col: str = "label",
    ) -> None:
        texts = [ex[text_col] for ex in samples]
        labels = [ex[label_col] for ex in samples]

        # Tokenisation en batch — padding + troncature uniforme
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids: torch.Tensor = encodings["input_ids"]
        self.attention_mask: torch.Tensor = encodings["attention_mask"]
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

        self._n = len(self.labels)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    @property
    def shape(self) -> tuple[int, int]:
        """Retourne (n_samples, seq_len)."""
        return tuple(self.input_ids.shape)  # type: ignore[return-value]
