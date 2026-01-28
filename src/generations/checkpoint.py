# Checkpoint management for model lineages
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Any


class CheckpointManager:
    """Manage checkpoints across generations of self-distillation.

    Saves and loads model states with metadata about lineage.
    """

    def __init__(self, base_dir: str | Path):
        """Initialize checkpoint manager.

        Args:
            base_dir: Base directory for all checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_file = self.base_dir / "lineage.json"
        self._load_lineage()

    def _load_lineage(self):
        """Load existing lineage data if present."""
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                self.lineage = json.load(f)
        else:
            self.lineage = {
                'generations': [],
                'created_at': datetime.now().isoformat(),
            }

    def _save_lineage(self):
        """Save lineage data to disk."""
        with open(self.lineage_file, 'w') as f:
            json.dump(self.lineage, f, indent=2)

    def generation_dir(self, generation: int) -> Path:
        """Get directory for a specific generation."""
        gen_dir = self.base_dir / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        return gen_dir

    def save_generation(
        self,
        generation: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        metrics: dict = None,
        config: dict = None,
        is_best: bool = False,
    ) -> Path:
        """Save a generation checkpoint.

        Args:
            generation: Generation number
            model: Model to save
            optimizer: Optional optimizer state
            metrics: Training metrics for this generation
            config: Training configuration used
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        gen_dir = self.generation_dir(generation)

        # Create checkpoint
        checkpoint = {
            'generation': generation,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'config': config or {},
            'timestamp': datetime.now().isoformat(),
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save checkpoint
        checkpoint_path = gen_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best separately
        if is_best:
            best_path = gen_dir / "best.pt"
            torch.save(checkpoint, best_path)

        # Update lineage
        gen_record = {
            'generation': generation,
            'parent': generation - 1 if generation > 0 else None,
            'checkpoint_path': str(checkpoint_path),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }

        # Update or append
        found = False
        for i, rec in enumerate(self.lineage['generations']):
            if rec['generation'] == generation:
                self.lineage['generations'][i] = gen_record
                found = True
                break

        if not found:
            self.lineage['generations'].append(gen_record)

        self._save_lineage()

        return checkpoint_path

    def load_generation(
        self,
        generation: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        load_best: bool = False,
    ) -> dict:
        """Load a generation checkpoint.

        Args:
            generation: Generation number to load
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            load_best: Whether to load best checkpoint instead of latest

        Returns:
            Checkpoint metadata dict
        """
        gen_dir = self.generation_dir(generation)

        filename = "best.pt" if load_best else "checkpoint.pt"
        checkpoint_path = gen_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'generation': checkpoint['generation'],
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'timestamp': checkpoint.get('timestamp'),
        }

    def get_latest_generation(self) -> int | None:
        """Get the latest saved generation number.

        Returns:
            Latest generation number or None if no generations saved
        """
        if not self.lineage['generations']:
            return None
        return max(g['generation'] for g in self.lineage['generations'])

    def get_lineage(self) -> dict:
        """Get full lineage data.

        Returns:
            Lineage dictionary with all generation records
        """
        return self.lineage

    def get_generation_metrics(self, generation: int) -> dict | None:
        """Get metrics for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Metrics dict or None if not found
        """
        for rec in self.lineage['generations']:
            if rec['generation'] == generation:
                return rec.get('metrics', {})
        return None
