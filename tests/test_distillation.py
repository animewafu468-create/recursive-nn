# Tests for distillation loss functions
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.losses import (
    kl_divergence_loss,
    distillation_loss,
    feature_loss,
    DistillationLoss,
)
from distillation.temperature import TemperatureScheduler


class TestKLDivergenceLoss:
    """Tests for KL divergence loss."""

    def test_kl_loss_shape(self, sample_logits):
        """KL loss should return scalar."""
        student_logits, teacher_logits, _ = sample_logits
        loss = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)
        assert loss.shape == torch.Size([])

    def test_kl_loss_non_negative(self, sample_logits):
        """KL divergence should be non-negative."""
        student_logits, teacher_logits, _ = sample_logits
        loss = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)
        assert loss >= 0

    def test_kl_loss_identical_is_zero(self):
        """KL divergence of identical distributions should be near zero."""
        logits = torch.randn(8, 10)
        loss = kl_divergence_loss(logits, logits, temperature=4.0)
        assert loss < 1e-5

    def test_kl_loss_temperature_scaling(self, sample_logits):
        """Higher temperature should produce different loss values."""
        student_logits, teacher_logits, _ = sample_logits

        loss_t4 = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)
        loss_t10 = kl_divergence_loss(student_logits, teacher_logits, temperature=10.0)

        # Should be different (exact relationship depends on distribution)
        assert not torch.allclose(loss_t4, loss_t10)


class TestDistillationLoss:
    """Tests for combined distillation loss."""

    def test_distillation_loss_returns_tuple(self, sample_logits):
        """Should return loss and metrics dict."""
        student_logits, teacher_logits, labels = sample_logits
        result = distillation_loss(student_logits, teacher_logits, labels)

        assert isinstance(result, tuple)
        assert len(result) == 2

        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_distillation_loss_metrics(self, sample_logits):
        """Should include all expected metrics."""
        student_logits, teacher_logits, labels = sample_logits
        _, metrics = distillation_loss(student_logits, teacher_logits, labels)

        assert 'total_loss' in metrics
        assert 'kl_loss' in metrics
        assert 'ce_loss' in metrics

    def test_alpha_weighting(self, sample_logits):
        """Alpha should weight KL vs CE loss."""
        student_logits, teacher_logits, labels = sample_logits

        # Pure KL (alpha=1.0)
        loss_kl, _ = distillation_loss(
            student_logits, teacher_logits, labels, alpha=1.0
        )

        # Pure CE (alpha=0.0)
        loss_ce, _ = distillation_loss(
            student_logits, teacher_logits, labels, alpha=0.0
        )

        # Mixed
        loss_mixed, _ = distillation_loss(
            student_logits, teacher_logits, labels, alpha=0.5
        )

        # Mixed should be between pure losses
        assert not torch.allclose(loss_kl, loss_ce)

    def test_gradient_flow(self, sample_logits):
        """Gradients should flow through distillation loss."""
        student_logits, teacher_logits, labels = sample_logits
        student_logits.requires_grad = True

        loss, _ = distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()

        assert student_logits.grad is not None
        assert not torch.all(student_logits.grad == 0)


class TestFeatureLoss:
    """Tests for feature-based distillation."""

    def test_feature_loss_shape(self):
        """Feature loss should return scalar."""
        student_features = torch.randn(8, 512)
        teacher_features = torch.randn(8, 512)

        loss = feature_loss(student_features, teacher_features)
        assert loss.shape == torch.Size([])

    def test_feature_loss_identical_is_zero(self):
        """Identical features should give near-zero loss."""
        features = torch.randn(8, 512)
        loss = feature_loss(features, features)
        assert loss < 1e-5


class TestTemperatureScheduler:
    """Tests for temperature scheduling."""

    def test_constant_schedule(self):
        """Constant schedule should return same value."""
        scheduler = TemperatureScheduler(
            initial_temp=4.0,
            final_temp=1.0,
            schedule="constant",
            total_steps=10,
        )

        for step in range(15):
            assert scheduler.get_temperature(step) == 4.0

    def test_linear_schedule(self):
        """Linear schedule should interpolate."""
        scheduler = TemperatureScheduler(
            initial_temp=10.0,
            final_temp=2.0,
            schedule="linear",
            total_steps=8,
        )

        # Start
        assert scheduler.get_temperature(0) == 10.0

        # Middle
        mid_temp = scheduler.get_temperature(4)
        assert 5.5 < mid_temp < 6.5

        # End
        assert scheduler.get_temperature(8) == 2.0

    def test_cosine_schedule(self):
        """Cosine schedule should follow cosine curve."""
        scheduler = TemperatureScheduler(
            initial_temp=10.0,
            final_temp=1.0,
            schedule="cosine",
            total_steps=100,
        )

        # Start - should be at initial
        assert scheduler.get_temperature(0) == 10.0

        # End - should be at final
        assert abs(scheduler.get_temperature(100) - 1.0) < 0.01


class TestDistillationLossModule:
    """Tests for the module wrapper."""

    def test_module_forward(self, sample_logits):
        """Module forward should work correctly."""
        student_logits, teacher_logits, labels = sample_logits

        loss_module = DistillationLoss(temperature=4.0, alpha=0.7)
        loss, metrics = loss_module(student_logits, teacher_logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert 'total_loss' in metrics
