# Tests for Transformer models
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.transformer import (
    PositionalEncoding,
    TransformerClassifier,
    TransformerForImages,
    TransformerSmall,
    VisionTransformerSmall,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        pe = PositionalEncoding(d_model=512)
        x = torch.randn(10, 2, 512)  # (seq_len, batch, d_model)
        
        out = pe(x)
        assert out.shape == x.shape

    def test_position_uniqueness(self):
        """Different positions should have different encodings."""
        pe = PositionalEncoding(d_model=512, max_len=100)
        
        # Check that position 0 and 50 have different encodings
        pos_0 = pe.pe[0, 0, :]
        pos_50 = pe.pe[50, 0, :]
        
        assert not torch.allclose(pos_0, pos_50)

    def test_dropout_effect(self):
        """Dropout should only apply in training."""
        pe = PositionalEncoding(d_model=512, dropout=0.5)
        x = torch.randn(10, 2, 512)
        
        pe.train()
        out1 = pe(x)
        out2 = pe(x)
        
        # Should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        pe.eval()
        out3 = pe(x)
        out4 = pe(x)
        
        # Should be same in eval
        assert torch.allclose(out3, out4)


class TestTransformerClassifier:
    """Tests for TransformerClassifier."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = TransformerClassifier(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=10,
        )
        
        # Input: (batch_size, seq_len)
        x = torch.randint(0, 1000, (4, 20))
        
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_seq_first(self):
        """Should handle (seq_len, batch) input."""
        model = TransformerClassifier(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=10,
        )
        
        # Input: (seq_len, batch_size)
        x = torch.randint(0, 1000, (20, 4))
        
        out = model(x)
        assert out.shape == (4, 10)

    def test_get_features(self):
        """Should return features before classifier."""
        model = TransformerClassifier(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=10,
        )
        
        x = torch.randint(0, 1000, (4, 20))
        features = model.get_features(x)
        
        assert features.shape == (4, 128)

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        model = TransformerClassifier(
            vocab_size=100,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=5,
        )
        
        x = torch.randint(0, 100, (2, 10))
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check some parameter has gradient
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in model.parameters()
        )
        assert has_grad


class TestTransformerForImages:
    """Tests for Vision Transformer."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = TransformerForImages(
            img_size=32,
            patch_size=4,
            d_model=256,
            num_classes=10,
        )
        
        # CIFAR-10 style input
        x = torch.randn(2, 3, 32, 32)
        
        out = model(x)
        assert out.shape == (2, 10)

    def test_different_patch_sizes(self):
        """Should handle different patch sizes."""
        model = TransformerForImages(
            img_size=32,
            patch_size=8,  # Larger patches
            d_model=256,
            num_classes=10,
        )
        
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        
        assert out.shape == (2, 10)

    def test_get_features(self):
        """Should return normalized features."""
        model = TransformerForImages(
            img_size=32,
            patch_size=4,
            d_model=256,
            num_classes=10,
        )
        
        x = torch.randn(2, 3, 32, 32)
        features = model.get_features(x)
        
        assert features.shape == (2, 256)

    def test_num_patches_calculation(self):
        """Should calculate number of patches correctly."""
        model = TransformerForImages(
            img_size=32,
            patch_size=4,
            d_model=256,
            num_classes=10,
        )
        
        # 32/4 = 8 patches per side, 8*8 = 64 patches
        assert model.num_patches == 64

    def test_patchify(self):
        """Patchify should create correct number of patches."""
        model = TransformerForImages(
            img_size=32,
            patch_size=4,
            d_model=256,
            num_classes=10,
        )
        
        x = torch.randn(2, 3, 32, 32)
        patches = model.patchify(x)
        
        # Should be (batch, num_patches, d_model)
        assert patches.shape == (2, 64, 256)


class TestModelFactories:
    """Tests for model factory functions."""

    def test_transformer_small(self):
        """TransformerSmall should create smaller model."""
        model = TransformerSmall(vocab_size=1000, num_classes=10)
        
        x = torch.randint(0, 1000, (4, 20))
        out = model(x)
        
        assert out.shape == (4, 10)
        assert model.d_model == 256  # Smaller than default

    def test_vision_transformer_small(self):
        """VisionTransformerSmall should work for CIFAR."""
        model = VisionTransformerSmall(num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        
        assert out.shape == (2, 10)

    def test_dropout_parameter(self):
        """Factory functions should respect dropout parameter."""
        model = TransformerSmall(dropout=0.5)
        
        # Check that dropout is set
        # Dropout modules should exist in the model
        has_dropout = any(
            isinstance(m, torch.nn.Dropout)
            for m in model.modules()
        )
        assert has_dropout
