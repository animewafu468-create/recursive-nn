# Metrics tracking and convergence detection for generations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MetricsTracker:
    """Track metrics across generations of self-distillation."""

    history: list[dict] = field(default_factory=list)

    def record(self, generation: int, metrics: dict):
        """Record metrics for a generation.

        Args:
            generation: Generation number
            metrics: Dictionary of metric values
        """
        self.history.append({
            'generation': generation,
            **metrics,
        })

    def get_metric_history(self, metric_name: str) -> list[float]:
        """Get history of a specific metric across generations.

        Args:
            metric_name: Name of metric to retrieve

        Returns:
            List of metric values in generation order
        """
        return [h.get(metric_name) for h in self.history if metric_name in h]

    def get_generation(self, generation: int) -> dict | None:
        """Get metrics for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Metrics dict or None
        """
        for h in self.history:
            if h['generation'] == generation:
                return h
        return None

    def get_best_generation(self, metric_name: str, higher_is_better: bool = True) -> int:
        """Find generation with best value for a metric.

        Args:
            metric_name: Metric to optimize
            higher_is_better: Whether higher values are better

        Returns:
            Generation number with best metric value
        """
        best_gen = 0
        best_val = float('-inf') if higher_is_better else float('inf')

        for h in self.history:
            if metric_name not in h:
                continue
            val = h[metric_name]
            if higher_is_better and val > best_val:
                best_val = val
                best_gen = h['generation']
            elif not higher_is_better and val < best_val:
                best_val = val
                best_gen = h['generation']

        return best_gen

    def compute_improvement(self, metric_name: str) -> list[float]:
        """Compute per-generation improvement for a metric.

        Args:
            metric_name: Metric to compute improvement for

        Returns:
            List of improvements (value[i] - value[i-1])
        """
        values = self.get_metric_history(metric_name)
        if len(values) < 2:
            return []
        return [values[i] - values[i - 1] for i in range(1, len(values))]


class ConvergenceDetector:
    """Detect when self-distillation has converged/plateaued."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        metric_name: str = 'val_acc',
        higher_is_better: bool = True,
    ):
        """Initialize convergence detector.

        Args:
            patience: Generations without improvement before stopping
            threshold: Minimum improvement to count as progress
            metric_name: Metric to monitor
            higher_is_better: Whether higher values are better
        """
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        self.best_value = float('-inf') if higher_is_better else float('inf')
        self.generations_without_improvement = 0
        self.history = []

    def update(self, value: float) -> dict:
        """Update with new metric value.

        Args:
            value: Latest metric value

        Returns:
            Status dict with convergence information
        """
        self.history.append(value)

        # Check for improvement
        improved = False
        if self.higher_is_better:
            if value > self.best_value + self.threshold:
                improved = True
                self.best_value = value
        else:
            if value < self.best_value - self.threshold:
                improved = True
                self.best_value = value

        if improved:
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        return {
            'current_value': value,
            'best_value': self.best_value,
            'improved': improved,
            'generations_without_improvement': self.generations_without_improvement,
            'should_stop': self.generations_without_improvement >= self.patience,
        }

    def should_stop(self) -> bool:
        """Check if training should stop due to convergence.

        Returns:
            True if converged/plateaued
        """
        return self.generations_without_improvement >= self.patience

    def get_trend(self, window: int = 3) -> str:
        """Analyze recent trend.

        Args:
            window: Number of recent values to analyze

        Returns:
            "improving", "degrading", or "plateau"
        """
        if len(self.history) < window:
            return "insufficient_data"

        recent = self.history[-window:]

        # Simple linear trend
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        avg_diff = sum(diffs) / len(diffs)

        if abs(avg_diff) < self.threshold:
            return "plateau"
        elif (self.higher_is_better and avg_diff > 0) or (not self.higher_is_better and avg_diff < 0):
            return "improving"
        else:
            return "degrading"
