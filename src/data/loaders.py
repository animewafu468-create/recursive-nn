# Dataset utilities for self-distillation experiments
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders.

    Args:
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        data_dir: Directory to store/load data
        augmentation: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Training transforms
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders.

    Args:
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        data_dir: Directory to store/load data

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# HaGRID gesture classes
HAGRID_CLASSES = [
    'call',      # 🤙 phone gesture
    'dislike',   # 👎 thumbs down
    'fist',      # ✊ closed fist
    'four',      # 4 fingers
    'like',      # 👍 thumbs up
    'mute',      # 🤫 finger on lips
    'ok',        # 👌 ok sign
    'one',       # ☝️ pointing up
    'palm',      # 🖐️ open palm / stop
    'peace',     # ✌️ peace sign
    'peace_inverted',  # ✌️ upside down peace
    'rock',      # 🤘 rock on
    'stop',      # ✋ stop gesture
    'stop_inverted',   # ✋ inverted stop
    'three',     # 3 fingers
    'three2',    # 3 fingers (alt)
    'two_up',    # 2 fingers up
    'two_up_inverted', # 2 fingers (alt)
]

HAGRID_EMOJIS = {
    'call': '🤙', 'dislike': '👎', 'fist': '✊', 'four': '4️⃣',
    'like': '👍', 'mute': '🤫', 'ok': '👌', 'one': '☝️',
    'palm': '🖐️', 'peace': '✌️', 'peace_inverted': '✌️', 'rock': '🤘',
    'stop': '✋', 'stop_inverted': '✋', 'three': '3️⃣', 'three2': '3️⃣',
    'two_up': '✌️', 'two_up_inverted': '✌️',
}


class HaGRIDDataset(torch.utils.data.Dataset):
    """HaGRID Hand Gesture Recognition Dataset.

    Download from: https://github.com/hukenovs/hagrid
    Or Kaggle: https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p

    Expected structure:
        data_dir/
        ├── train/
        │   ├── call/
        │   ├── dislike/
        │   ├── fist/
        │   └── ...
        └── test/
            ├── call/
            ├── dislike/
            └── ...
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
    ):
        from pathlib import Path

        self.root = Path(root) / split
        self.transform = transform
        self.classes = HAGRID_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Gather all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root}. "
                f"Download HaGRID and organize into {root}/train/<gesture>/ folders."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_hagrid_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data/hagrid",
    image_size: int = 224,
    augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Get HaGRID hand gesture data loaders.

    Args:
        batch_size: Batch size
        num_workers: Number of workers
        data_dir: Path to HaGRID dataset
        image_size: Resize images to this size
        augmentation: Apply training augmentation

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # ImageNet normalization (HaGRID uses real photos)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = HaGRIDDataset(data_dir, split='train', transform=train_transform)
    test_dataset = HaGRIDDataset(data_dir, split='test', transform=test_transform)

    print(f"HaGRID loaded: {len(train_dataset)} train, {len(test_dataset)} test images")
    print(f"Classes: {len(HAGRID_CLASSES)} gestures")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_augmented_transforms() -> transforms.Compose:
    """Get strong augmentation transforms for Noisy Student training.

    Returns:
        Composed transforms with RandAugment-style augmentation
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5),
    ])
