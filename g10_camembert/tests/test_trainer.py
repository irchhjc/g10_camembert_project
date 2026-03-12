"""Tests unitaires pour le module training (mocked)."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from g10_camembert.utils.metrics import compute_metrics, generalization_gap


class TestGetOptimizerParams:
    def test_no_decay_params_excluded(self):
        """Vérifie que bias et LayerNorm sont exclus du weight decay."""
        from g10_camembert.training.trainer import get_optimizer_params

        model = MagicMock()
        model.named_parameters.return_value = [
            ("layer.weight", torch.nn.Parameter(torch.randn(4, 4))),
            ("layer.bias", torch.nn.Parameter(torch.randn(4))),
            ("LayerNorm.weight", torch.nn.Parameter(torch.randn(4))),
        ]

        groups = get_optimizer_params(model, weight_decay=1e-3)
        assert len(groups) == 2

        # Groupe avec weight decay
        decay_group = groups[0]
        no_decay_group = groups[1]

        assert decay_group["weight_decay"] == 1e-3
        assert no_decay_group["weight_decay"] == 0.0

    def test_groups_cover_all_params(self):
        """Vérifie que tous les paramètres sont couverts."""
        from g10_camembert.training.trainer import get_optimizer_params

        model = MagicMock()
        p1 = torch.nn.Parameter(torch.randn(4, 4))
        p2 = torch.nn.Parameter(torch.randn(4))
        p3 = torch.nn.Parameter(torch.randn(4))

        model.named_parameters.return_value = [
            ("layer.weight", p1),
            ("layer.bias", p2),
            ("LayerNorm.weight", p3),
        ]

        groups = get_optimizer_params(model, weight_decay=1e-4)
        all_params = groups[0]["params"] + groups[1]["params"]
        assert len(all_params) == 3


class TestMetricsIntegration:
    def test_pipeline_metrics(self):
        """Test du pipeline complet de métriques."""
        labels = [0, 0, 0, 1, 1, 1]
        preds = [0, 0, 1, 1, 1, 0]

        m = compute_metrics(labels, preds)
        assert "f1_macro" in m
        assert "accuracy" in m
        assert 0.0 <= m["f1_macro"] <= 1.0

        gap = generalization_gap(m["f1_macro"], m["f1_macro"] - 0.05)
        assert abs(gap["gap"] - 0.05) < 1e-6
