# Tests for model architectures
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import MLP, ResNet18, ResNet34


class TestMLP:
    """Tests for MLP model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = MLP(input_dim=784, hidden_dims=[256, 128], num_classes=10)
        x = torch.randn(8, 784)

        output = model(x)
        assert output.shape == (8, 10)

    def test_forward_with_images(self):
        """Should handle image inputs (flattens automatically)."""
        model = MLP(input_dim=3072, hidden_dims=[512, 256], num_classes=10)
        x = torch.randn(8, 3, 32, 32)  # CIFAR-10 style

        output = model(x)
        assert output.shape == (8, 10)

    def test_get_features(self):
        """Should return intermediate features."""
        model = MLP(input_dim=784, hidden_dims=[256, 128], num_classes=10)
        x = torch.randn(8, 784)

        features = model.get_features(x)
        assert features.shape == (8, 128)

    def test_dropout(self):
        """Dropout should affect training vs eval."""
        model = MLP(input_dim=100, num_classes=10, dropout_rate=0.5)
        x = torch.randn(8, 100)

        model.train()
        out1 = model(x)
        out2 = model(x)

        # With dropout, outputs should differ
        assert not torch.allclose(out1, out2)

        model.eval()
        out3 = model(x)
        out4 = model(x)

        # Without dropout, outputs should be same
        assert torch.allclose(out3, out4)

    def test_count_parameters(self):
        """Should count parameters correctly."""
        model = MLP(input_dim=100, hidden_dims=[50], num_classes=10)
        param_count = model.count_parameters()

        # 100*50 + 50 (first layer) + 50*10 + 10 (output)
        expected = 100 * 50 + 50 + 50 * 10 + 10 + 50  # +50 for batchnorm
        assert param_count > 0


class TestResNet18:
    """Tests for ResNet-18 model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = ResNet18(num_classes=10)
        x = torch.randn(4, 3, 32, 32)  # CIFAR-10

        output = model(x)
        assert output.shape == (4, 10)

    def test_forward_different_batch_sizes(self):
        """Should handle various batch sizes."""
        model = ResNet18(num_classes=10)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_get_features(self):
        """Should return intermediate features."""
        model = ResNet18(num_classes=10)
        x = torch.randn(4, 3, 32, 32)

        features = model.get_features(x)
        assert features.shape == (4, 512)  # ResNet-18 has 512 features

    def test_dropout_effect(self):
        """Dropout should affect training vs eval."""
        model = ResNet18(num_classes=10, dropout_rate=0.5)
        x = torch.randn(4, 3, 32, 32)

        model.train()
        out1 = model(x)
        out2 = model(x)

        # With dropout, outputs differ
        assert not torch.allclose(out1, out2)

        model.eval()
        out3 = model(x)
        out4 = model(x)

        # Without dropout, outputs same
        assert torch.allclose(out3, out4)

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        model = ResNet18(num_classes=10)
        x = torch.randn(4, 3, 32, 32, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestResNet34:
    """Tests for ResNet-34 model."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        model = ResNet34(num_classes=100)
        x = torch.randn(4, 3, 32, 32)

        output = model(x)
        assert output.shape == (4, 100)

    def test_more_parameters_than_resnet18(self):
        """ResNet-34 should have more parameters than ResNet-18."""
        resnet18 = ResNet18(num_classes=10)
        resnet34 = ResNet34(num_classes=10)

        params18 = resnet18.count_parameters()
        params34 = resnet34.count_parameters()

        assert params34 > params18
