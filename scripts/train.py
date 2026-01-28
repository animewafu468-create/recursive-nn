#!/usr/bin/env python3
# Main training entry point for self-distillation experiments
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from models import ResNet18, MLP
from data import get_cifar10_loaders, get_mnist_loaders, get_hagrid_loaders
from generations import GenerationManager


def get_model_factory(config: DictConfig):
    """Get model factory function based on config."""
    model_name = config.model.name.lower()
    num_classes = config.model.num_classes
    dropout_rate = config.distillation.get('dropout_rate', 0.0)

    if model_name == 'resnet18':
        return lambda: ResNet18(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'resnet34':
        from models import ResNet34
        return lambda: ResNet34(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'mlp':
        return lambda: MLP(
            input_dim=784 if config.data.name == 'mnist' else 3072,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_data_loaders(config: DictConfig):
    """Get data loaders based on config."""
    data_config = config.data
    data_name = data_config.name.lower()

    if data_name == 'cifar10':
        return get_cifar10_loaders(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            augmentation=data_config.augmentation,
        )
    elif data_name == 'mnist':
        return get_mnist_loaders(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
        )
    elif data_name == 'hagrid':
        return get_hagrid_loaders(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            image_size=data_config.get('image_size', 224),
            augmentation=data_config.get('augmentation', True),
        )
    else:
        raise ValueError(f"Unknown dataset: {data_name}")


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(config: DictConfig):
    """Main training function."""
    print("=" * 60)
    print("Recursive Self-Distillation Neural Network")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Model factory
    model_factory = get_model_factory(config)
    sample_model = model_factory()
    print(f"\nModel: {config.model.name}")
    print(f"Parameters: {sample_model.count_parameters():,}")
    del sample_model

    # Convert config to dict for manager
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Generation manager
    manager = GenerationManager(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict,
        checkpoint_dir=config.checkpoint.save_dir,
        device=device,
    )

    # Run evolution
    print("\nStarting self-distillation evolution...")
    summary = manager.run_evolution(
        num_generations=config.generations.max_generations,
        early_stop=True,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Generations trained: {summary['total_generations']}")
    print(f"  Best generation: {summary['best_generation']}")
    print(f"  Best validation accuracy: {summary['best_val_acc']:.2f}%")

    if summary['improvement_history']:
        print(f"\nImprovement per generation:")
        for i, imp in enumerate(summary['improvement_history']):
            print(f"  Gen {i} → Gen {i+1}: {imp:+.2f}%")

    return summary


if __name__ == "__main__":
    main()
