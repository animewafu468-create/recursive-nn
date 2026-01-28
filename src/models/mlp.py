# Simple MLP for testing self-distillation on MNIST/CIFAR-10
import torch
import torch.nn as nn
from models.base import BaseModel


class MLP(BaseModel):
    """Multi-layer perceptron for simple classification tasks.

    Used for initial testing of self-distillation before moving to
    more complex architectures.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = None,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        self._feature_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (for image inputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        features = self.features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.features(x)
