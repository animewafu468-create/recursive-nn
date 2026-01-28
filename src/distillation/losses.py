# Loss functions for knowledge distillation
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute KL divergence between softened distributions.

    KL(teacher || student) with temperature scaling.
    Multiplied by T^2 to maintain gradient magnitude.

    Args:
        student_logits: Raw logits from student [B, C]
        teacher_logits: Raw logits from teacher [B, C]
        temperature: Softmax temperature (higher = softer distributions)

    Returns:
        Scalar loss value
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence with T^2 scaling for gradient correction
    return F.kl_div(
        log_soft_student,
        soft_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict]:
    """Combined soft + hard target distillation loss.

    L = alpha * KL(teacher || student) * T^2 + (1 - alpha) * CE(labels, student)

    Args:
        student_logits: Raw logits from student [B, C]
        teacher_logits: Raw logits from teacher [B, C] (no gradient)
        labels: Ground truth class labels [B]
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss (1-alpha for CE loss)

    Returns:
        total_loss: Combined loss scalar
        metrics: Dictionary with component losses
    """
    # Soft target loss (distillation)
    kl_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)

    # Hard target loss (classification)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    metrics = {
        'total_loss': total_loss.item(),
        'kl_loss': kl_loss.item(),
        'ce_loss': ce_loss.item(),
    }

    return total_loss, metrics


def feature_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Feature-based distillation loss (FitNets style).

    Matches intermediate feature representations between teacher and student.

    Args:
        student_features: Features from student [B, D]
        teacher_features: Features from teacher [B, D']

    Returns:
        MSE loss between normalized features
    """
    # Normalize features
    student_norm = F.normalize(student_features, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_features, p=2, dim=-1)

    return F.mse_loss(student_norm, teacher_norm)


class DistillationLoss(nn.Module):
    """Module wrapper for distillation loss with configurable parameters."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        use_features: bool = False,
        feature_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_features = use_features
        self.feature_weight = feature_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined distillation loss."""
        total_loss, metrics = distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            self.temperature,
            self.alpha,
        )

        if self.use_features and student_features is not None and teacher_features is not None:
            feat_loss = feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.feature_weight * feat_loss
            metrics['feature_loss'] = feat_loss.item()

        return total_loss, metrics
