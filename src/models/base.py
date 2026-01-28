# Base model interface for self-distillation
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models in the self-distillation system.

    Provides common interface for forward pass and feature extraction.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        pass

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features (for feature-based distillation).

        Default implementation returns None. Override in subclasses for
        feature matching distillation.
        """
        return None

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self):
        """Reset all parameters to initial values."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
