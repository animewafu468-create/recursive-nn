# Generation lifecycle manager for recursive self-distillation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from pathlib import Path

from generations.checkpoint import CheckpointManager
from generations.metrics import MetricsTracker, ConvergenceDetector
from distillation.trainer import DistillationTrainer


class GenerationManager:
    """Manage the full lifecycle of recursive self-distillation.

    Trains multiple generations where each generation learns from the previous.
    Tracks lineage, detects convergence, and saves checkpoints.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: str | Path = "./checkpoints",
        device: torch.device = None,
    ):
        """Initialize the generation manager.

        Args:
            model_factory: Callable that creates a new model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Full configuration dictionary
            checkpoint_dir: Directory for checkpoints
            device: Device to train on
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Managers
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.metrics_tracker = MetricsTracker()

        # Generation config
        gen_config = config.get('generations', {})
        self.max_generations = gen_config.get('max_generations', 5)

        # Convergence detection
        self.convergence_detector = ConvergenceDetector(
            patience=gen_config.get('plateau_patience', 3),
            threshold=gen_config.get('plateau_threshold', 0.001),
            metric_name='val_acc',
            higher_is_better=True,
        )

        # Current state
        self.generations: list[nn.Module] = []
        self.current_generation = 0

    def train_generation(self, generation: int) -> dict:
        """Train a single generation.

        Args:
            generation: Generation number (0 = base model)

        Returns:
            Training metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training Generation {generation}")
        print(f"{'='*60}")

        # Create new student model
        student = self.model_factory()
        student.to(self.device)

        if generation == 0:
            # Generation 0: Train from hard labels only
            metrics = self._train_from_scratch(student)
        else:
            # Subsequent generations: Use previous generation as teacher
            teacher = self.generations[-1]
            teacher.to(self.device)
            teacher.eval()

            trainer = DistillationTrainer(
                student=student,
                teacher=teacher,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=self.config,
                device=self.device,
            )

            metrics = trainer.train()

        # Store generation
        self.generations.append(student)

        # Record metrics
        self.metrics_tracker.record(generation, metrics)

        # Save checkpoint
        self.checkpoint_manager.save_generation(
            generation=generation,
            model=student,
            metrics=metrics,
            config=self.config,
            is_best=True,
        )

        return metrics

    def _train_from_scratch(self, model: nn.Module) -> dict:
        """Train generation 0 from hard labels only.

        Args:
            model: Model to train

        Returns:
            Training metrics
        """
        train_config = self.config.get('training', {})
        epochs = train_config.get('epochs', 100)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config.get('learning_rate', 0.1),
            momentum=train_config.get('momentum', 0.9),
            weight_decay=train_config.get('weight_decay', 1e-4),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total

            # Validate
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_acc = 100. * correct / total

            scheduler.step()

            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}%"
                )

        # Restore best
        if best_state:
            model.load_state_dict(best_state)

        return {
            'best_val_acc': best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
        }

    def run_evolution(
        self,
        num_generations: int = None,
        early_stop: bool = True,
    ) -> dict:
        """Run full self-distillation evolution.

        Args:
            num_generations: Max generations (overrides config)
            early_stop: Whether to stop on convergence

        Returns:
            Summary of all generations
        """
        max_gen = num_generations or self.max_generations

        for gen in range(max_gen):
            metrics = self.train_generation(gen)

            # Check convergence
            status = self.convergence_detector.update(metrics['best_val_acc'])

            print(f"\nGeneration {gen} complete:")
            print(f"  Best Val Acc: {metrics['best_val_acc']:.2f}%")
            print(f"  Improvement: {'Yes' if status['improved'] else 'No'}")
            print(f"  Trend: {self.convergence_detector.get_trend()}")

            if early_stop and status['should_stop']:
                print(f"\nConverged after {gen + 1} generations (no improvement for {self.convergence_detector.patience} generations)")
                break

        # Summary
        best_gen = self.metrics_tracker.get_best_generation('best_val_acc', higher_is_better=True)
        best_metrics = self.metrics_tracker.get_generation(best_gen)

        summary = {
            'total_generations': len(self.generations),
            'best_generation': best_gen,
            'best_val_acc': best_metrics['best_val_acc'] if best_metrics else 0,
            'improvement_history': self.metrics_tracker.compute_improvement('best_val_acc'),
            'all_metrics': self.metrics_tracker.history,
        }

        print(f"\n{'='*60}")
        print("Evolution Complete")
        print(f"{'='*60}")
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Best Generation: {best_gen}")
        print(f"Best Val Acc: {summary['best_val_acc']:.2f}%")

        return summary

    def load_generation(self, generation: int) -> nn.Module:
        """Load a specific generation from checkpoint.

        Args:
            generation: Generation number to load

        Returns:
            Loaded model
        """
        model = self.model_factory()
        self.checkpoint_manager.load_generation(generation, model)
        return model

    def resume(self, from_generation: int = None) -> int:
        """Resume training from a checkpoint.

        Args:
            from_generation: Generation to resume from (latest if None)

        Returns:
            Generation to continue from
        """
        if from_generation is None:
            from_generation = self.checkpoint_manager.get_latest_generation()

        if from_generation is None:
            return 0

        # Load all previous generations
        for gen in range(from_generation + 1):
            model = self.load_generation(gen)
            self.generations.append(model)

            # Restore metrics
            gen_metrics = self.checkpoint_manager.get_generation_metrics(gen)
            if gen_metrics:
                self.metrics_tracker.record(gen, gen_metrics)
                self.convergence_detector.update(gen_metrics.get('best_val_acc', 0))

        self.current_generation = from_generation + 1
        return self.current_generation
