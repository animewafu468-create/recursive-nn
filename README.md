# Recursive Self-Improving Neural Network via Self-Distillation

A general-purpose self-distillation system where neural networks iteratively train improved versions of themselves. Each generation uses soft targets from the previous generation combined with hard labels, progressively refining knowledge through "dark knowledge" transfer.

## 🎯 Key Features

- **Born-Again Networks**: Student and teacher share identical architecture
- **Soft Target Distillation**: Transfers rich probability distributions at high temperature
- **Noisy Student Training**: Adds noise (dropout, augmentation) for robustness
- **EMA Teacher Updates**: Exponential moving average for stable distillation
- **Automatic Convergence Detection**: Stops when improvements plateau
- **Multi-Architecture Support**: MLP, ResNet, Vision Transformer

## 📁 Project Structure

```
recursive-nn/
├── configs/
│   └── default.yaml          # Hydra configuration
├── src/
│   ├── models/               # Model architectures
│   │   ├── mlp.py           # Multi-layer perceptron
│   │   ├── resnet.py        # ResNet-18/34 for CIFAR
│   │   ├── transformer.py   # Transformer for sequences/images
│   │   └── stochastic_depth.py  # Noisy Student enhancement
│   ├── distillation/        # Core distillation logic
│   │   ├── losses.py        # KL divergence, combined loss
│   │   ├── trainer.py       # Single-generation trainer
│   │   ├── ema.py          # EMA teacher updates
│   │   └── temperature.py   # Temperature scheduling
│   ├── generations/         # Generation management
│   │   ├── manager.py       # Lifecycle management
│   │   ├── checkpoint.py    # Save/load lineages
│   │   └── metrics.py       # Convergence detection
│   └── data/               # Dataset utilities
│       └── loaders.py       # CIFAR-10, MNIST loaders
├── scripts/
│   ├── train.py            # Main training entry point
│   └── evaluate.py         # Compare generations
├── tests/                  # Comprehensive test suite
└── notebooks/
    └── experiments.ipynb   # Interactive exploration
```

## 🚀 Quick Start

### Installation

```bash
cd recursive-nn
pip install -e ".[dev]"
```

### Train a Model

```bash
# Train 5 generations on CIFAR-10 with ResNet-18
python scripts/train.py \
    model=resnet18 \
    data=cifar10 \
    training.epochs=100 \
    generations.max_generations=5

# Quick test on MNIST with MLP
python scripts/train.py \
    model=mlp \
    data=mnist \
    training.epochs=10 \
    generations.max_generations=3
```

### Evaluate Generations

```bash
python scripts/evaluate.py \
    --checkpoint-dir ./checkpoints \
    --model resnet18 \
    --dataset cifar10
```

## 🔬 How It Works

### Core Algorithm

```
Generation 0: Train base model on hard labels
Generation 1: Train new model on (soft targets from Gen0 + hard labels)
Generation 2: Train new model on (soft targets from Gen1 + hard labels)
...
Generation N: Each generation typically improves over the last
```

### Distillation Loss

```python
L = α * KL(student||teacher) * T² + (1-α) * CE(labels, student)
```

Where:
- `α` (alpha): Weight for distillation vs. hard labels (default: 0.7)
- `T` (temperature): Softens probability distributions (default: 4.0)
- Higher temperature → softer targets → more "dark knowledge"

### Key Techniques

1. **Soft Target Distillation** (Hinton et al., 2015)
   - Teacher produces softened softmax outputs
   - Student learns from probability distributions, not just argmax
   - Captures class similarities in "dark knowledge"

2. **Born-Again Networks** (Furlanello et al., 2018)
   - Student has identical architecture to teacher
   - Surprisingly outperforms teacher despite same capacity
   - Multiple generations compound improvements

3. **Noisy Student** (Xie et al., 2020)
   - Add dropout noise during student training
   - Strong data augmentation (RandAugment)
   - Stochastic depth for deeper networks

## ⚙️ Configuration

All parameters controlled via `configs/default.yaml`:

```yaml
# Model
model:
  name: "resnet18"        # resnet18, resnet34, mlp, vit_small
  num_classes: 10

# Data
data:
  name: "cifar10"         # cifar10, mnist
  batch_size: 128
  augmentation: true      # RandAugment for Noisy Student

# Distillation
distillation:
  temperature: 4.0        # Softmax temperature
  alpha: 0.7              # Weight for distillation loss
  noisy_student: true     # Enable noise/augmentation
  dropout_rate: 0.1       # Dropout for regularization

# Generations
generations:
  max_generations: 5      # Maximum iterations
  plateau_threshold: 0.001  # Min improvement to continue
  plateau_patience: 3     # Generations without improvement before stop
```

Override any parameter from command line:

```bash
python scripts/train.py distillation.temperature=10.0 generations.max_generations=10
```

## 📊 Expected Results

### CIFAR-10 with ResNet-18

| Generation | Val Accuracy | Improvement |
|------------|--------------|-------------|
| Gen 0 (baseline) | ~92.0% | - |
| Gen 1 | ~93.5% | +1.5% |
| Gen 2 | ~94.2% | +0.7% |
| Gen 3 | ~94.5% | +0.3% |
| Gen 4+ | Plateau | ~0% |

**Total improvement**: 2-3% over baseline with same architecture!

### MNIST with 2-Layer MLP

| Generation | Val Accuracy |
|------------|--------------|
| Gen 0 | ~98.0% |
| Gen 1 | ~98.4% |
| Gen 2 | ~98.5% |

## 🔧 Advanced Features

### EMA Teacher

For more stable distillation:

```python
from distillation import EMATeacher

ema_teacher = EMATeacher(student_model, decay=0.999)

# During training
for batch in dataloader:
    train_step(student_model, batch)
    ema_teacher.update(student_model)
    
# Use EMA model as teacher
teacher_logits = ema_teacher.model(inputs)
```

### Temperature Scheduling

Schedule temperature across generations:

```python
from distillation import TemperatureScheduler

scheduler = TemperatureScheduler(
    initial_temp=20.0,   # High temp early for dark knowledge
    final_temp=1.0,      # Low temp later for sharp predictions
    schedule="cosine",   # "constant", "linear", "cosine", "step"
    total_steps=10,      # Number of generations
)

temp = scheduler.get_temperature(generation_num)
```

### Stochastic Depth

For deeper ResNets (Noisy Student):

```python
from models.stochastic_depth import apply_stochastic_depth

model = ResNet18(num_classes=10)
model = apply_stochastic_depth(model, drop_prob=0.2)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_distillation.py

# Run with coverage
pytest --cov=src
```

## 📓 Interactive Notebooks

Explore the system interactively:

```bash
cd notebooks
jupyter notebook experiments.ipynb
```

Includes:
- Temperature visualization
- Quick MNIST experiments
- Confidence analysis
- Feature visualization

## 📚 References

1. **Born-Again Neural Networks** (Furlanello et al., 2018)
   - https://arxiv.org/abs/1805.04770

2. **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
   - https://arxiv.org/abs/1503.02531

3. **Self-Training with Noisy Student** (Xie et al., 2020)
   - https://arxiv.org/abs/1911.04252

4. **Mean Teachers are Better Role Models** (Tarvaninen & Valpola, 2017)
   - https://arxiv.org/abs/1703.01780

5. **Deep Networks with Stochastic Depth** (Huang et al., 2016)
   - https://arxiv.org/abs/1603.09382

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Feature-based distillation (FitNets, attention transfer)
- Multi-teacher distillation
- Data-efficient training (semi-supervised)
- Distributed training support
- Additional architectures ( EfficientNet, ConvNeXt)

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Papers With Code for distillation implementations
- The self-distillation research community
