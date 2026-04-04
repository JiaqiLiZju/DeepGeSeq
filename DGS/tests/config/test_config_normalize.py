"""Unit tests for configuration normalization."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from DGS.Config import normalize_config, ConfigManager


class TestConfigNormalization(unittest.TestCase):
    """Test cases for config normalization."""
    def test_legacy_optimizer_and_criterion_are_normalized(self):
        """Test legacy optimizer and criterion are normalized."""
        config = {
            "train": {
                "optimizer": {
                    "type": "Adam",
                    "lr": 1e-3,
                    "weight_decay": 0.1,
                },
                "criterion": {
                    "type": "MSELoss",
                },
            }
        }

        normalized, notes = normalize_config(config)
        self.assertIn("params", normalized["train"]["optimizer"])
        self.assertEqual(normalized["train"]["optimizer"]["params"]["lr"], 1e-3)
        self.assertEqual(normalized["train"]["optimizer"]["params"]["weight_decay"], 0.1)
        self.assertIn("params", normalized["train"]["criterion"])
        self.assertEqual(normalized["train"]["criterion"]["params"], {})
        self.assertTrue(notes)

    def test_normalization_is_idempotent(self):
        """Test normalization is idempotent."""
        config = {
            "train": {
                "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
                "criterion": {"type": "MSELoss", "params": {}},
            }
        }
        normalized_once, notes_once = normalize_config(config)
        normalized_twice, notes_twice = normalize_config(normalized_once)
        self.assertEqual(normalized_once, normalized_twice)
        self.assertEqual(notes_once, [])
        self.assertEqual(notes_twice, [])

    def test_config_manager_load_applies_normalization(self):
        """Test config manager load applies normalization."""
        config_manager = ConfigManager()
        normalized = config_manager.load_config(
            {
                "train": {
                    "optimizer": {"type": "Adam", "lr": 0.002},
                    "criterion": {"type": "MSELoss"},
                }
            }
        )
        self.assertEqual(normalized["train"]["optimizer"]["params"]["lr"], 0.002)
        self.assertIn("params", normalized["train"]["criterion"])
        self.assertTrue(config_manager.get_compat_notes())


if __name__ == "__main__":
    unittest.main()
