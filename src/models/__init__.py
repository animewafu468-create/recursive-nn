# Model definitions for self-distillation
from models.base import BaseModel
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from models.transformer import (
    TransformerClassifier,
    TransformerForImages,
    TransformerSmall,
    VisionTransformerSmall,
)

__all__ = [
    "BaseModel",
    "MLP",
    "ResNet18",
    "ResNet34",
    "TransformerClassifier",
    "TransformerForImages",
    "TransformerSmall",
    "VisionTransformerSmall",
]
