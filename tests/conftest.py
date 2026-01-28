# pytest configuration and fixtures
import pytest
import torch


@pytest.fixture
def device():
    """Provide test device."""
    return torch.device('cpu')


@pytest.fixture
def random_seed():
    """Set deterministic random seed for tests."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    batch_size = 8
    num_classes = 10

    # CIFAR-10 style images
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, num_classes, (batch_size,))

    return images, labels


@pytest.fixture
def sample_logits():
    """Create sample logits for distillation testing."""
    batch_size = 8
    num_classes = 10

    teacher_logits = torch.randn(batch_size, num_classes)
    student_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    return student_logits, teacher_logits, labels


@pytest.fixture
def simple_config():
    """Create simple test configuration."""
    return {
        'model': {
            'name': 'resnet18',
            'num_classes': 10,
        },
        'data': {
            'name': 'cifar10',
            'batch_size': 8,
            'num_workers': 0,
            'augmentation': False,
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
        },
        'distillation': {
            'temperature': 4.0,
            'alpha': 0.7,
            'noisy_student': False,
            'dropout_rate': 0.0,
        },
        'generations': {
            'max_generations': 2,
            'plateau_threshold': 0.001,
            'plateau_patience': 2,
        },
        'seed': 42,
    }
