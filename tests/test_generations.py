# Tests for generation management
import pytest
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import ResNet18, MLP
from generations.checkpoint import CheckpointManager
from generations.metrics import MetricsTracker, ConvergenceDetector


class TestCheckpointManager:
    """Tests for checkpoint management."""

    def test_save_and_load_generation(self, device):
        """Should save and load generation correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Create and save model
            model = MLP(input_dim=100, num_classes=10)
            original_state = {k: v.clone() for k, v in model.state_dict().items()}

            metrics = {'val_acc': 85.5, 'train_loss': 0.5}
            manager.save_generation(0, model, metrics=metrics)

            # Reset model weights
            model.reset_parameters()

            # Load and verify
            loaded = manager.load_generation(0, model)

            for key in original_state:
                assert torch.allclose(
                    model.state_dict()[key],
                    original_state[key]
                )

            assert loaded['generation'] == 0
            assert loaded['metrics']['val_acc'] == 85.5

    def test_lineage_tracking(self):
        """Should track lineage across generations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            for gen in range(3):
                model = MLP(input_dim=100, num_classes=10)
                manager.save_generation(gen, model, metrics={'gen': gen})

            lineage = manager.get_lineage()
            assert len(lineage['generations']) == 3

            # Check parent relationships
            assert lineage['generations'][0]['parent'] is None
            assert lineage['generations'][1]['parent'] == 0
            assert lineage['generations'][2]['parent'] == 1

    def test_get_latest_generation(self):
        """Should return latest generation number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            assert manager.get_latest_generation() is None

            for gen in range(3):
                model = MLP(input_dim=100, num_classes=10)
                manager.save_generation(gen, model)

            assert manager.get_latest_generation() == 2


class TestMetricsTracker:
    """Tests for metrics tracking."""

    def test_record_and_retrieve(self):
        """Should record and retrieve metrics."""
        tracker = MetricsTracker()

        tracker.record(0, {'val_acc': 80.0, 'train_loss': 0.5})
        tracker.record(1, {'val_acc': 85.0, 'train_loss': 0.4})

        gen1 = tracker.get_generation(1)
        assert gen1['val_acc'] == 85.0
        assert gen1['train_loss'] == 0.4

    def test_metric_history(self):
        """Should track metric history across generations."""
        tracker = MetricsTracker()

        tracker.record(0, {'val_acc': 80.0})
        tracker.record(1, {'val_acc': 85.0})
        tracker.record(2, {'val_acc': 87.0})

        history = tracker.get_metric_history('val_acc')
        assert history == [80.0, 85.0, 87.0]

    def test_best_generation(self):
        """Should find best generation for a metric."""
        tracker = MetricsTracker()

        tracker.record(0, {'val_acc': 80.0, 'val_loss': 0.5})
        tracker.record(1, {'val_acc': 90.0, 'val_loss': 0.3})
        tracker.record(2, {'val_acc': 85.0, 'val_loss': 0.4})

        # Best accuracy (higher is better)
        assert tracker.get_best_generation('val_acc', higher_is_better=True) == 1

        # Best loss (lower is better)
        assert tracker.get_best_generation('val_loss', higher_is_better=False) == 1

    def test_compute_improvement(self):
        """Should compute per-generation improvement."""
        tracker = MetricsTracker()

        tracker.record(0, {'val_acc': 80.0})
        tracker.record(1, {'val_acc': 85.0})
        tracker.record(2, {'val_acc': 86.0})

        improvements = tracker.compute_improvement('val_acc')
        assert len(improvements) == 2
        assert improvements[0] == 5.0
        assert improvements[1] == 1.0


class TestConvergenceDetector:
    """Tests for convergence detection."""

    def test_initial_state(self):
        """Should not stop initially."""
        detector = ConvergenceDetector(patience=3, threshold=0.001)
        assert not detector.should_stop()

    def test_detect_improvement(self):
        """Should detect improvement."""
        detector = ConvergenceDetector(
            patience=3,
            threshold=0.001,
            higher_is_better=True,
        )

        status = detector.update(80.0)
        assert status['improved']

        status = detector.update(85.0)
        assert status['improved']

        status = detector.update(85.001)  # Below threshold
        assert not status['improved']

    def test_plateau_detection(self):
        """Should detect plateau after patience exceeded."""
        detector = ConvergenceDetector(
            patience=2,
            threshold=0.5,
            higher_is_better=True,
        )

        detector.update(80.0)  # Initial
        detector.update(80.1)  # No significant improvement
        detector.update(80.2)  # Still no improvement

        assert detector.should_stop()

    def test_trend_analysis(self):
        """Should analyze recent trend."""
        detector = ConvergenceDetector(
            patience=5,
            threshold=0.1,
            higher_is_better=True,
        )

        # Improving trend
        for val in [80.0, 81.0, 82.0, 83.0]:
            detector.update(val)

        assert detector.get_trend(window=3) == "improving"

        # Now plateau
        for val in [83.0, 83.05, 83.02]:
            detector.update(val)

        assert detector.get_trend(window=3) == "plateau"

    def test_lower_is_better(self):
        """Should work with metrics where lower is better."""
        detector = ConvergenceDetector(
            patience=3,
            threshold=0.01,
            higher_is_better=False,
        )

        status = detector.update(0.5)
        assert status['improved']

        status = detector.update(0.4)
        assert status['improved']

        status = detector.update(0.5)  # Got worse
        assert not status['improved']
