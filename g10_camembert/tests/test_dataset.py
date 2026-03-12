"""Tests unitaires pour le module data."""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestBalancedSubsample:
    def test_balanced_output(self):
        from g10_camembert.data.loader import balanced_subsample

        # Jeu de données synthétique déséquilibré
        examples = [{"review": f"text {i}", "label": i % 2} for i in range(100)]
        result = balanced_subsample(examples, n_per_class=20)

        labels = [ex["label"] for ex in result]
        assert labels.count(0) == 20
        assert labels.count(1) == 20

    def test_reproducibility(self):
        from g10_camembert.data.loader import balanced_subsample

        examples = [{"review": f"text {i}", "label": i % 2} for i in range(100)]
        r1 = balanced_subsample(examples, n_per_class=10, seed=42)
        r2 = balanced_subsample(examples, n_per_class=10, seed=42)

        assert [ex["review"] for ex in r1] == [ex["review"] for ex in r2]

    def test_different_seeds(self):
        from g10_camembert.data.loader import balanced_subsample

        examples = [{"review": f"text {i}", "label": i % 2} for i in range(200)]
        r1 = balanced_subsample(examples, n_per_class=30, seed=1)
        r2 = balanced_subsample(examples, n_per_class=30, seed=2)

        texts1 = set(ex["review"] for ex in r1)
        texts2 = set(ex["review"] for ex in r2)
        # Très probable que les deux soient différents
        assert texts1 != texts2


class TestAllocinéDataset:
    @pytest.fixture
    def mock_tokenizer(self):
        """Tokenizer simulé pour les tests."""
        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.randint(0, 1000, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
        }
        return tok

    def test_len(self, mock_tokenizer):
        from g10_camembert.data.dataset import AllocinéDataset

        samples = [{"review": f"critique {i}", "label": i % 2} for i in range(4)]
        ds = AllocinéDataset(samples, mock_tokenizer, max_length=16)
        assert len(ds) == 4

    def test_getitem_keys(self, mock_tokenizer):
        from g10_camembert.data.dataset import AllocinéDataset

        samples = [{"review": "bon film", "label": 1}]
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        ds = AllocinéDataset(samples, mock_tokenizer, max_length=16)
        item = ds[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_label_dtype(self, mock_tokenizer):
        from g10_camembert.data.dataset import AllocinéDataset

        samples = [{"review": f"film {i}", "label": i % 2} for i in range(4)]
        ds = AllocinéDataset(samples, mock_tokenizer, max_length=16)
        assert ds.labels.dtype == torch.long
