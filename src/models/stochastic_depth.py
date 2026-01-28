# Stochastic Depth (Drop Path) for ResNet - Noisy Student enhancement
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    
    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    
    During training, randomly drops entire residual paths with probability
    `drop_prob`. During inference, acts as identity (scaling by survival prob).
    
    This is a key component of Noisy Student training, making the student
    more robust than the teacher.
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize DropPath.
        
        Args:
            drop_prob: Probability of dropping the path (0 = always keep)
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path during training.
        
        Args:
            x: Input tensor
            
        Returns:
            Input tensor (possibly zeroed out)
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        # Generate random mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize: 0 or 1 with prob keep_prob
        
        # Scale output by 1/keep_prob to maintain expected value
        output = x.div(self.keep_prob) * random_tensor
        return output
        
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.4f}'


class StochasticDepthBlock(nn.Module):
    """Residual block with stochastic depth.
    
    Wraps a residual block and randomly skips it during training.
    """

    def __init__(self, block: nn.Module, drop_prob: float = 0.0):
        """Initialize stochastic depth block.
        
        Args:
            block: Residual block to wrap
            drop_prob: Probability of dropping this block
        """
        super().__init__()
        self.block = block
        self.drop_path = DropPath(drop_prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with stochastic depth.
        
        The residual connection is:
        - During training: sometimes skip the block entirely
        - During inference: always use the block
        """
        # Apply block, then drop path on the residual
        residual = self.block(x)
        residual = self.drop_path(residual)
        
        # Add residual to input (assuming block includes its own shortcut)
        # Note: This assumes block returns the full output including shortcut
        return residual


def apply_stochastic_depth(
    model: nn.Module,
    drop_prob: float = 0.1,
    linear_schedule: bool = True,
) -> nn.Module:
    """Apply stochastic depth to all residual blocks in a model.
    
    Uses linear schedule: earlier layers have lower drop probability.
    
    Args:
        model: Model to modify (e.g., ResNet)
        drop_prob: Maximum drop probability (for last layers)
        linear_schedule: If True, linearly increase drop prob with depth
        
    Returns:
        Modified model (modified in-place)
    """
    # Find all Sequential layers that contain residual blocks
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Count blocks in this sequential
            num_blocks = len(module)
            
            # Apply stochastic depth to each block
            for i, block in enumerate(module):
                if linear_schedule and num_blocks > 1:
                    # Linear schedule: first block has 0 drop, last has drop_prob
                    block_drop_prob = drop_prob * i / (num_blocks - 1)
                else:
                    block_drop_prob = drop_prob
                    
                if block_drop_prob > 0:
                    # Wrap block with stochastic depth
                    module[i] = StochasticDepthBlock(block, block_drop_prob)
                    
    return model


class StochasticDepthResNet(nn.Module):
    """ResNet with stochastic depth enabled.
    
    Convenience wrapper that creates a ResNet and applies
    stochastic depth to all residual blocks.
    """

    def __init__(
        self,
        base_resnet: nn.Module,
        drop_prob: float = 0.1,
        linear_schedule: bool = True,
    ):
        """Initialize ResNet with stochastic depth.
        
        Args:
            base_resnet: Base ResNet model
            drop_prob: Maximum drop probability
            linear_schedule: Use linear schedule for drop probability
        """
        super().__init__()
        self.model = apply_stochastic_depth(
            base_resnet,
            drop_prob=drop_prob,
            linear_schedule=linear_schedule,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features if base model supports it."""
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(x)
        raise NotImplementedError("Base model doesn't support feature extraction")


# Utility functions for computing expected depth
def compute_expected_depth(model: nn.Module, drop_prob: float = 0.1) -> float:
    """Compute expected depth of model with stochastic depth.
    
    Args:
        model: Model with stochastic depth blocks
        drop_prob: Drop probability
        
    Returns:
        Expected depth (number of blocks that will be active on average)
    """
    total_blocks = 0
    stochastic_blocks = 0
    
    for module in model.modules():
        if isinstance(module, StochasticDepthBlock):
            stochastic_blocks += 1
        elif isinstance(module, nn.Sequential):
            total_blocks += len(module)
            
    if total_blocks == 0:
        return 0
        
    # Expected number of active blocks
    expected = total_blocks * (1 - drop_prob / 2)  # Average of linear schedule
    return expected
