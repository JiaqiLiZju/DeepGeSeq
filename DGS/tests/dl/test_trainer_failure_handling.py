"""Unit tests for trainer failure accounting and checkpoint loading behavior."""

import tempfile
import unittest
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.DL.Trainer import Trainer


class _FailingModel(torch.nn.Module):
    """Model that raises on every forward pass."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, _x):
        raise RuntimeError("intentional forward failure")


class TestTrainerFailureHandling(unittest.TestCase):
    """Validate robust error handling in train/validate loops."""

    def setUp(self):
        dataset = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
        self.loader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.device = torch.device("cpu")

    def test_train_epoch_raises_when_all_batches_fail(self):
        """All failed train batches should raise a clear error."""
        model = _FailingModel()
        trainer = Trainer(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device=self.device,
        )

        with self.assertRaisesRegex(RuntimeError, "All training batches failed"):
            trainer.train_epoch(self.loader, epoch=0)

    def test_validate_raises_when_all_batches_fail(self):
        """All failed validation batches should raise a clear error."""
        model = _FailingModel()
        trainer = Trainer(
            model=model,
            criterion=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            device=self.device,
        )

        with self.assertRaisesRegex(RuntimeError, "All validation batches failed"):
            trainer.validate(self.loader)

    def test_load_checkpoint_can_skip_optimizer_state(self):
        """Evaluation workflows can load model weights without optimizer restore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"

            model_src = torch.nn.Linear(2, 1)
            trainer_src = Trainer(
                model=model_src,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.Adam(model_src.parameters(), lr=1e-3),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_src.save_checkpoint(ckpt_path)

            model_dst = torch.nn.Linear(2, 1)
            trainer_dst = Trainer(
                model=model_dst,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(model_dst.parameters(), lr=1e-2),
                device=self.device,
                checkpoint_dir=tmp_dir,
            )
            trainer_dst.load_checkpoint(ckpt_path, load_optimizer=False)

            for src_param, dst_param in zip(model_src.parameters(), trainer_dst.model.parameters()):
                self.assertTrue(torch.allclose(src_param, dst_param))


if __name__ == "__main__":
    unittest.main()

