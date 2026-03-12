"""Tests unitaires pour le module utils.metrics."""

import pytest
import numpy as np
from g10_camembert.metrics import (
    compute_metrics,
    generalization_gap,
    compute_sharpness,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_all_wrong(self):
        labels = [0, 0, 1, 1]
        preds = [1, 1, 0, 0]
        m = compute_metrics(labels, preds)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["f1_macro"] == pytest.approx(0.0)

    def test_balanced_random(self):
        labels = [0, 1, 0, 1]
        preds = [0, 0, 1, 1]
        m = compute_metrics(labels, preds)
        assert 0.0 <= m["f1_macro"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0


class TestGeneralizationGap:
    def test_no_gap(self):
        g = generalization_gap(0.9, 0.9)
        assert g["gap"] == pytest.approx(0.0)
        assert g["gap_pct"] == pytest.approx(0.0)

    def test_positive_gap(self):
        g = generalization_gap(0.95, 0.90)
        assert g["gap"] == pytest.approx(0.05)
        assert g["gap_pct"] == pytest.approx(5.0 / 0.95 * 100, rel=1e-3)

    def test_negative_gap(self):
        """Gap négatif possible sur sous-ensembles."""
        g = generalization_gap(0.85, 0.93)
        assert g["gap"] < 0

    def test_zero_train_f1(self):
        g = generalization_gap(0.0, 0.5)
        assert g["gap_pct"] == pytest.approx(0.0)


class TestComputeSharpness:
    def test_flat_landscape(self):
        """Loss constante → sharpness nulle."""
        base = 0.5
        perturbed = [0.5, 0.5, 0.5, 0.5]
        assert compute_sharpness(base, perturbed) == pytest.approx(0.0)

    def test_sharp_landscape(self):
        base = 0.3
        perturbed = [0.8, 0.9, 0.8, 0.9]
        s = compute_sharpness(base, perturbed)
        assert s == pytest.approx(0.55, rel=1e-3)

    def test_symmetric_perturbation(self):
        base = 0.5
        perturbed = [0.6, 0.5, 0.4]  # symétrique autour de base
        s = compute_sharpness(base, perturbed)
        # avg(|0.1|, |0.0|, |-0.1|) = 0.1/3 * 2 ≈ 0.0667
        assert s == pytest.approx(0.2 / 3, rel=1e-3)
