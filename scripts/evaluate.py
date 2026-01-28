#!/usr/bin/env python3
# Evaluate and compare generations of self-distilled models
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ResNet18, MLP
from data import get_cifar10_loaders
from generations import CheckpointManager


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    all_probs = torch.cat(all_probs, dim=0)

    # Compute confidence metrics
    max_probs = all_probs.max(dim=1)[0]
    entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-10), dim=1)

    return {
        'accuracy': 100. * correct / total,
        'loss': total_loss / total,
        'avg_confidence': max_probs.mean().item(),
        'avg_entropy': entropy.mean().item(),
    }


def compare_generations(
    checkpoint_dir: str,
    model_factory,
    test_loader: DataLoader,
    device: torch.device,
):
    """Compare all generations in a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing generation checkpoints
        model_factory: Function to create model instances
        test_loader: Test data loader
        device: Device to evaluate on
    """
    ckpt_manager = CheckpointManager(checkpoint_dir)
    lineage = ckpt_manager.get_lineage()

    print("\n" + "=" * 70)
    print("Generation Comparison")
    print("=" * 70)
    print(f"{'Gen':<5} {'Accuracy':<12} {'Loss':<10} {'Confidence':<12} {'Entropy':<10}")
    print("-" * 70)

    results = []

    for gen_record in lineage['generations']:
        gen = gen_record['generation']

        model = model_factory()
        ckpt_manager.load_generation(gen, model, load_best=True)

        metrics = evaluate_model(model, test_loader, device)
        results.append({'generation': gen, **metrics})

        print(
            f"{gen:<5} "
            f"{metrics['accuracy']:<12.2f} "
            f"{metrics['loss']:<10.4f} "
            f"{metrics['avg_confidence']:<12.4f} "
            f"{metrics['avg_entropy']:<10.4f}"
        )

    print("-" * 70)

    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Generation: {best['generation']} with {best['accuracy']:.2f}% accuracy")

    # Compute improvements
    if len(results) > 1:
        print("\nImprovements:")
        for i in range(1, len(results)):
            prev = results[i - 1]['accuracy']
            curr = results[i]['accuracy']
            print(f"  Gen {i-1} → Gen {i}: {curr - prev:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate self-distilled models")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "mlp"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data
    if args.dataset == "cifar10":
        _, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size,
            num_workers=4,
            augmentation=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model factory
    if args.model == "resnet18":
        model_factory = lambda: ResNet18(num_classes=10)
    elif args.model == "mlp":
        model_factory = lambda: MLP(input_dim=3072, num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Compare
    compare_generations(
        checkpoint_dir=args.checkpoint_dir,
        model_factory=model_factory,
        test_loader=test_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
