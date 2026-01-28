# Knowledge distillation components
from distillation.losses import distillation_loss, kl_divergence_loss, feature_loss
from distillation.temperature import TemperatureScheduler
from distillation.trainer import DistillationTrainer
from distillation.ema import EMATeacher, EMADistillationTrainer

__all__ = [
    "distillation_loss",
    "kl_divergence_loss",
    "feature_loss",
    "TemperatureScheduler",
    "DistillationTrainer",
    "EMATeacher",
    "EMADistillationTrainer",
]
