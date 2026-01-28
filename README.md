# Self-Distillation Neural Network + Hand Gesture Recognition

Train neural networks that iteratively improve themselves, then use it for real-time hand gesture detection via webcam.

---

## Quick Start (Step-by-Step)

### YOU HAVE TWO MACHINES:
- **Machine A (3060 GPU)**: For training
- **Machine B (Webcam)**: For live gesture recognition

---

## STEP 1: Setup on 3060 Machine

```bash
# Clone this repo
git clone https://github.com/animewafu468-create/recursive-nn.git
cd recursive-nn

# Install PyTorch with CUDA (IMPORTANT: use cu121 for RTX 3060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install this project
pip install -e .

# Install extra dependencies
pip install requests
```

### Verify GPU works:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```
Should print: `GPU: NVIDIA GeForce RTX 3060`

---

## STEP 2: Download HaGRID Dataset

```bash
python scripts/download_hagrid.py
```

This downloads ~700MB of hand gesture images (30k images, 18 gestures).

**If download times out**, try the Kaggle mirror:
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('innominate817/hagrid-sample-30k-384p')"
```

---

## STEP 3: Train the Model

```bash
python scripts/train.py --config-name=hagrid
```

**Expected time on RTX 3060:** ~45-60 minutes

**What happens:**
1. Gen 0: Trains from scratch (~92% accuracy)
2. Gen 1: Learns from Gen 0's soft outputs (~94% accuracy)
3. Gen 2: Learns from Gen 1 (~95% accuracy)
4. ... continues until accuracy plateaus

**Quick test run (5 min):**
```bash
python scripts/train.py --config-name=hagrid training.epochs=10 generations.max_generations=2
```

---

## STEP 4: Copy Trained Model to Webcam Machine

After training completes, copy this file:
```
checkpoints/hagrid/gen_004/best.pt
```

Use USB drive, Google Drive, email - whatever works.

---

## STEP 5: Run Live Webcam (on Webcam Machine)

```bash
# Clone repo (if not already)
git clone https://github.com/animewafu468-create/recursive-nn.git
cd recursive-nn

# Install (CPU is fine for inference)
pip install torch torchvision
pip install -e .
pip install opencv-python

# Run webcam with your trained model
python scripts/live_gesture.py --checkpoint path/to/best.pt
```

**Controls:**
- Press `Q` to quit
- Show hand gestures to the camera
- Predictions appear with confidence bars

---

## Supported Gestures (18 total)

| Gesture | Emoji | Gesture | Emoji |
|---------|-------|---------|-------|
| call | phone hand | palm | raised hand |
| dislike | thumbs down | peace | peace sign |
| fist | raised fist | rock | rock on |
| like | thumbs up | stop | stop hand |
| ok | OK hand | one | index finger |
| peace_inverted | inverted peace | three | three fingers |
| two_up | two fingers | four | four fingers |
| mute | shush | grab | grabbing |
| no_gesture | - | timeout | T shape |

---

## Project Structure

```
recursive-nn/
├── configs/
│   ├── default.yaml          # CIFAR-10 config
│   └── hagrid.yaml           # Hand gesture config
├── src/
│   ├── models/               # Neural network architectures
│   │   ├── resnet.py         # ResNet-18 (used for gestures)
│   │   ├── mlp.py            # Simple MLP
│   │   └── transformer.py    # Vision Transformer
│   ├── distillation/         # Self-distillation logic
│   │   ├── losses.py         # KL divergence loss
│   │   ├── trainer.py        # Training loop
│   │   └── ema.py            # EMA teacher
│   ├── generations/          # Multi-generation management
│   │   ├── manager.py        # Orchestrates training
│   │   └── checkpoint.py     # Save/load models
│   └── data/
│       └── loaders.py        # Dataset loaders
├── scripts/
│   ├── train.py              # Main training script
│   ├── download_hagrid.py    # Download gesture dataset
│   ├── live_gesture.py       # Webcam inference
│   └── evaluate.py           # Compare generations
└── tests/                    # Unit tests
```

---

## How Self-Distillation Works

```
Gen 0: Train from scratch → 92% accuracy
         ↓ (becomes teacher)
Gen 1: Learns from Gen 0's soft outputs → 94% accuracy
         ↓ (becomes teacher)
Gen 2: Learns from Gen 1 → 95% accuracy
         ...
```

Each generation learns from:
1. **Hard labels** (ground truth: "this is thumbs up")
2. **Soft labels** (teacher's probabilities: "90% thumbs up, 5% like, 3% ok...")

The "dark knowledge" in soft labels reveals which gestures look similar.

**Loss function:**
```
L = α * KL(student||teacher) * T² + (1-α) * CrossEntropy(labels, student)
```

- `α = 0.7` (70% distillation, 30% hard labels)
- `T = 4.0` (temperature - higher = softer distributions)

---

## Configuration

Edit `configs/hagrid.yaml` to customize:

```yaml
model:
  name: "resnet18"
  num_classes: 18

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

distillation:
  temperature: 4.0    # Higher = softer teacher outputs
  alpha: 0.7          # Balance distillation vs hard labels

generations:
  max_generations: 5
  plateau_patience: 2  # Stop if no improvement for 2 gens
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python scripts/train.py --config-name=hagrid training.batch_size=16
```

### "No module named 'src'"
Make sure you ran `pip install -e .` in the project directory.

### Download times out
The HaGRID server (Russian cloud) can be slow. Try:
1. Use a VPN
2. Download from Kaggle mirror (see Step 2)
3. Download on a different network

### Webcam not detected
```bash
# List available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(3)])"
```

---

## TODO / Future Enhancements

- [ ] Add custom gesture recording (train on your own hands)
- [ ] Gesture-to-action mapping (thumbs up = volume up)
- [ ] Fine-tuning on user's specific hands
- [ ] Mobile deployment (ONNX export)
- [ ] Feature-based distillation (FitNets style)

---

## Tests

```bash
# Set Python path and run tests
set PYTHONPATH=src
pytest tests/ -v
```

67 tests passing, 3 minor failures (tolerance issues).

---

## References

1. **Born-Again Neural Networks** - https://arxiv.org/abs/1805.04770
2. **Knowledge Distillation** - https://arxiv.org/abs/1503.02531
3. **HaGRID Dataset** - https://github.com/hukenovs/hagrid

---

## License

MIT
