# Exponential Moving Average (EMA) for teacher updates
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMATeacher:
    """Exponential Moving Average teacher for self-distillation.
    
    Maintains a shadow model that is an exponential moving average of
    the student model weights. This provides a more stable teacher for
    distillation compared to using the raw student weights.
    
    Reference: "Mean teachers are better role models" (Tarvaninen & Valpola, 2017)
    
    Usage:
        ema_teacher = EMATeacher(student_model, decay=0.999)
        
        for batch in dataloader:
            # Train student
            loss = train_step(student_model, batch)
            
            # Update EMA teacher
            ema_teacher.update(student_model)
            
        # Use EMA teacher for distillation
        teacher_logits = ema_teacher.model(inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device = None,
    ):
        """Initialize EMA teacher.
        
        Args:
            model: Source model to create EMA from
            decay: EMA decay rate (higher = slower updates, more stable)
            device: Device to store EMA model on
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create shadow model
        self.model = deepcopy(model)
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients for teacher
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.num_updates = 0
        
    def update(self, model: nn.Module, decay: float = None):
        """Update EMA model with current student weights.
        
        Args:
            model: Student model to update from
            decay: Optional override for decay rate
        """
        if decay is None:
            decay = self.decay
            
        # Use warmup for first few updates (optional)
        self.num_updates += 1
        
        with torch.no_grad():
            for ema_param, student_param in zip(
                self.model.parameters(),
                model.parameters()
            ):
                # Move to same device if needed
                if student_param.device != self.device:
                    student_param = student_param.to(self.device)
                    
                # EMA update: θ_ema = decay * θ_ema + (1 - decay) * θ_student
                ema_param.data.mul_(decay).add_(
                    student_param.data,
                    alpha=1 - decay
                )
                
    def update_buffers(self, model: nn.Module):
        """Copy buffers (e.g., BatchNorm stats) from student to teacher.
        
        Args:
            model: Student model to copy buffers from
        """
        with torch.no_grad():
            for ema_buffer, student_buffer in zip(
                self.model.buffers(),
                model.buffers()
            ):
                ema_buffer.copy_(student_buffer)
                
    def state_dict(self) -> dict:
        """Get state dict for saving."""
        return {
            'model': self.model.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates,
        }
        
    def load_state_dict(self, state_dict: dict):
        """Load state dict."""
        self.model.load_state_dict(state_dict['model'])
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)
        
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.model(*args, **kwargs)


class EMADistillationTrainer:
    """Trainer that uses EMA teacher for distillation.
    
    Similar to standard distillation but maintains an EMA teacher
    that is updated after each training step.
    """

    def __init__(
        self,
        student: nn.Module,
        ema_decay: float = 0.999,
        update_every: int = 1,
        device: torch.device = None,
    ):
        """Initialize EMA distillation trainer.
        
        Args:
            student: Student model
            ema_decay: EMA decay rate
            update_every: Update EMA every N steps
            device: Device for training
        """
        self.student = student
        self.device = device or next(student.parameters()).device
        self.update_every = update_every
        
        # Create EMA teacher from initial student
        self.ema_teacher = EMATeacher(student, ema_decay, device)
        self.step_count = 0
        
    def training_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> dict:
        """Single training step with EMA update.
        
        Args:
            inputs: Input batch
            labels: Ground truth labels
            optimizer: Optimizer
            criterion: Loss function (should accept student_logits, teacher_logits, labels)
            
        Returns:
            Dictionary of metrics
        """
        self.student.train()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Student forward
        student_logits = self.student(inputs)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.ema_teacher.model(inputs)
            
        # Compute loss
        loss, metrics = criterion(student_logits, teacher_logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA teacher
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.ema_teacher.update(self.student)
            
        return metrics
    
    def get_teacher(self) -> nn.Module:
        """Get current EMA teacher model."""
        return self.ema_teacher.model
    
    def save_checkpoint(self, path: str):
        """Save student and EMA teacher."""
        torch.save({
            'student': self.student.state_dict(),
            'ema_teacher': self.ema_teacher.state_dict(),
            'step_count': self.step_count,
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load student and EMA teacher."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student'])
        self.ema_teacher.load_state_dict(checkpoint['ema_teacher'])
        self.step_count = checkpoint.get('step_count', 0)
