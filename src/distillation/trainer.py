# Self-distillation training loop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

from distillation.losses import distillation_loss
from distillation.temperature import TemperatureScheduler


class DistillationTrainer:
    """Trainer for single-generation knowledge distillation.

    Trains a student model using soft targets from a teacher model
    combined with hard labels.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device = None,
    ):
        """Initialize the distillation trainer.

        Args:
            student: Student model to train
            teacher: Teacher model (frozen)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move models to device
        self.student.to(self.device)
        self.teacher.to(self.device)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Distillation parameters
        dist_config = config.get('distillation', {})
        self.temperature = dist_config.get('temperature', 4.0)
        self.alpha = dist_config.get('alpha', 0.7)
        self.noisy_student = dist_config.get('noisy_student', True)
        self.dropout_rate = dist_config.get('dropout_rate', 0.1)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        train_config = self.config.get('training', {})
        return optim.SGD(
            self.student.parameters(),
            lr=train_config.get('learning_rate', 0.1),
            momentum=train_config.get('momentum', 0.9),
            weight_decay=train_config.get('weight_decay', 1e-4),
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler from config."""
        train_config = self.config.get('training', {})
        epochs = train_config.get('epochs', 100)
        scheduler_type = train_config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[epochs // 3, 2 * epochs // 3],
                gamma=0.1,
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.student.train()

        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Student forward
            student_logits = self.student(inputs)

            # Compute loss
            loss, metrics = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=self.temperature,
                alpha=self.alpha,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += metrics['total_loss'] * inputs.size(0)
            total_kl += metrics['kl_loss'] * inputs.size(0)
            total_ce += metrics['ce_loss'] * inputs.size(0)

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

        n = len(self.train_loader.dataset)
        return {
            'train_loss': total_loss / n,
            'train_kl_loss': total_kl / n,
            'train_ce_loss': total_ce / n,
            'train_acc': 100. * correct / total,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the student model.

        Returns:
            Dictionary of validation metrics
        """
        self.student.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Student forward
            student_logits = self.student(inputs)

            # Only CE loss for validation
            loss = nn.functional.cross_entropy(student_logits, labels)

            total_loss += loss.item() * inputs.size(0)

            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        n = len(self.val_loader.dataset)
        return {
            'val_loss': total_loss / n,
            'val_acc': 100. * correct / total,
        }

    def train(
        self,
        epochs: int = None,
        callback: Callable[[int, dict], None] = None,
    ) -> dict:
        """Full training loop.

        Args:
            epochs: Number of epochs (overrides config)
            callback: Optional callback(epoch, metrics) after each epoch

        Returns:
            Final metrics dictionary
        """
        epochs = epochs or self.config.get('training', {}).get('epochs', 100)
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            metrics['lr'] = self.scheduler.get_last_lr()[0]

            # Track best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}

            # Callback
            if callback:
                callback(epoch, metrics)

            # Log
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

        # Restore best model
        if best_state:
            self.student.load_state_dict(best_state)

        return {
            'best_val_acc': best_val_acc,
            'final_train_acc': train_metrics['train_acc'],
            'final_val_acc': val_metrics['val_acc'],
        }
