# Tests for EMA teacher
import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import MLP
from distillation.ema import EMATeacher, EMADistillationTrainer


class TestEMATeacher:
    """Tests for EMA teacher."""

    def test_ema_initialization(self, device):
        """EMA teacher should initialize from student."""
        student = MLP(input_dim=100, num_classes=10)
        ema = EMATeacher(student, decay=0.999, device=device)

        # EMA model should have same architecture
        assert ema.model is not None
        
        # EMA parameters should be close to student (but not exactly same due to deepcopy)
        for p_ema, p_student in zip(ema.model.parameters(), student.parameters()):
            assert torch.allclose(p_ema, p_student)

    def test_ema_no_grad(self, device):
        """EMA model should not require gradients."""
        student = MLP(input_dim=100, num_classes=10)
        ema = EMATeacher(student, device=device)

        for param in ema.model.parameters():
            assert not param.requires_grad

    def test_ema_update(self, device):
        """EMA update should blend student and EMA weights."""
        student = MLP(input_dim=100, num_classes=10)
        ema = EMATeacher(student, decay=0.9, device=device)

        # Get initial EMA weight
        initial_ema_weight = list(ema.model.parameters())[0].clone()

        # Modify student
        for param in student.parameters():
            param.data += 1.0

        # Update EMA
        ema.update(student)

        # EMA should be closer to new student than before
        new_ema_weight = list(ema.model.parameters())[0]
        
        # Should not be identical to either
        assert not torch.allclose(new_ema_weight, initial_ema_weight)

    def test_ema_state_dict(self, device, tmp_path):
        """EMA state dict should save and load correctly."""
        student = MLP(input_dim=100, num_classes=10)
        ema = EMATeacher(student, decay=0.999, device=device)

        # Save
        state = ema.state_dict()
        assert 'model' in state
        assert 'decay' in state
        assert state['decay'] == 0.999

        # Modify and restore
        for param in ema.model.parameters():
            param.data.zero_()

        ema.load_state_dict(state)

        # Verify restored
        for p_ema, p_student in zip(ema.model.parameters(), student.parameters()):
            assert torch.allclose(p_ema, p_student)


class TestEMADistillationTrainer:
    """Tests for EMA distillation trainer."""

    def test_trainer_initialization(self, device):
        """Trainer should create EMA teacher from student."""
        student = MLP(input_dim=100, num_classes=10)
        trainer = EMADistillationTrainer(
            student=student,
            ema_decay=0.999,
            device=device,
        )

        assert trainer.ema_teacher is not None
        assert trainer.step_count == 0

    def test_training_step_updates_ema(self, device):
        """Training step should update EMA teacher."""
        student = MLP(input_dim=10, num_classes=5)
        student.to(device)

        trainer = EMADistillationTrainer(
            student=student,
            ema_decay=0.9,
            update_every=1,
            device=device,
        )

        # Get initial teacher weight
        initial_teacher_weight = list(trainer.ema_teacher.model.parameters())[0].clone()

        # Create optimizer and simple criterion
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        
        def criterion(student_logits, teacher_logits, labels):
            loss = nn.functional.cross_entropy(student_logits, labels)
            return loss, {'loss': loss.item()}

        # Training step
        inputs = torch.randn(4, 10, device=device)
        labels = torch.randint(0, 5, (4,), device=device)

        metrics = trainer.training_step(inputs, labels, optimizer, criterion)

        # Step count should increment
        assert trainer.step_count == 1

        # Teacher should have updated
        new_teacher_weight = list(trainer.ema_teacher.model.parameters())[0]
        assert not torch.allclose(new_teacher_weight, initial_teacher_weight)

    def test_teacher_produces_logits(self, device):
        """EMA teacher should produce valid logits."""
        student = MLP(input_dim=10, num_classes=5)
        trainer = EMADistillationTrainer(student, device=device)

        inputs = torch.randn(4, 10, device=device)
        
        with torch.no_grad():
            teacher_logits = trainer.ema_teacher.model(inputs)

        assert teacher_logits.shape == (4, 5)
