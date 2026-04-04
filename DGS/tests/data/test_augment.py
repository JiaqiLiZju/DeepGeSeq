"""Tests for NvTK Data augmentation module."""

import pytest
import numpy as np
import torch

from DGS.Data.Augment import (
    AugmentationConfig,
    SequenceAugmentor,
    ReverseComplement,
    SequenceShift,
    RandomMutation,
    RandomRC,
    RandomJitter
)

# Test data fixtures
@pytest.fixture
def sequence_data():
    """Create test sequence data."""
    # Create one-hot encoded sequence data
    # Shape: (batch_size, channels, length)
    X = np.zeros((2, 4, 10))
    # Add some patterns
    X[0, 0, :5] = 1  # AAAAA in first sequence
    X[1, 1, 5:] = 1  # CCCCC in second sequence
    return X

@pytest.fixture
def label_data():
    """Create test label data."""
    return np.array([[1, 0], [0, 1]])

# Test configuration
def test_augmentation_config():
    """Test augmentation configuration."""
    config = AugmentationConfig(
        rc_prob=0.7,
        max_shift=5,
        shift_prob=0.4,
        mutation_rate=0.1,
        protect_sites=[0, 1],
        seed=42
    )
    
    assert config.rc_prob == 0.7
    assert config.max_shift == 5
    assert config.shift_prob == 0.4
    assert config.mutation_rate == 0.1
    assert config.protect_sites == [0, 1]
    assert config.seed == 42

# Test individual augmentations
class TestReverseComplement:
    """Test reverse complement augmentation."""
    
    def test_rc_transform(self, sequence_data):
        """Test reverse complement transformation."""
        rc = ReverseComplement(prob=1.0)  # Always apply RC
        X_aug = rc(sequence_data)
        
        # Check shape preserved
        assert X_aug.shape == sequence_data.shape
        
        # Check reverse complement correct
        np.testing.assert_array_equal(
            X_aug[0, :, :],
            sequence_data[0, ::-1, ::-1]
        )
        
    def test_rc_probability(self, sequence_data):
        """Test RC probability."""
        rc = ReverseComplement(prob=0.0)  # Never apply RC
        X_aug = rc(sequence_data)
        np.testing.assert_array_equal(X_aug, sequence_data)

class TestSequenceShift:
    """Test sequence shifting augmentation."""
    
    def test_shift_transform(self, sequence_data):
        """Test shift transformation."""
        shifter = SequenceShift(max_shift=2, prob=1.0)
        X_aug = shifter(sequence_data)
        
        # Check shape preserved
        assert X_aug.shape == sequence_data.shape
        
        # Test with specific shift
        X_shifted = shifter._shift_sequence(sequence_data, shift=1, pad_value=0.25)
        assert X_shifted.shape == sequence_data.shape
        # Check padding
        assert np.all(X_shifted[:, :, 0] == 0.25)
        
    def test_shift_probability(self, sequence_data):
        """Test shift probability."""
        shifter = SequenceShift(max_shift=2, prob=0.0)
        X_aug = shifter(sequence_data)
        np.testing.assert_array_equal(X_aug, sequence_data)

class TestRandomMutation:
    """Test random mutation augmentation."""
    
    def test_mutation_transform(self, sequence_data):
        """Test mutation transformation."""
        mutator = RandomMutation(rate=1.0)  # Mutate all positions
        X_aug = mutator(sequence_data)
        
        # Check shape preserved
        assert X_aug.shape == sequence_data.shape
        # Check some mutations occurred
        assert not np.array_equal(X_aug, sequence_data)
        
    def test_protect_sites(self, sequence_data):
        """Test protected sites not mutated."""
        protect_sites = [0, 1]
        mutator = RandomMutation(rate=1.0, protect_sites=protect_sites)
        X_aug = mutator(sequence_data)
        
        # Check protected sites unchanged
        for site in protect_sites:
            np.testing.assert_array_equal(
                X_aug[:, :, site],
                sequence_data[:, :, site]
            )

# Test full augmentation pipeline
class TestSequenceAugmentor:
    """Test sequence augmentor pipeline."""
    
    @pytest.fixture
    def config(self):
        """Build a default configuration fixture for augmentation tests."""
        return AugmentationConfig(
            rc_prob=0.5,
            max_shift=2,
            shift_prob=0.5,
            mutation_rate=0.1,
            seed=42
        )
    
    def test_augmentor_transform(self, sequence_data, label_data, config):
        """Test full augmentation pipeline."""
        augmentor = SequenceAugmentor(config)
        
        # Test without labels
        X_aug = augmentor(sequence_data)
        assert X_aug.shape == sequence_data.shape
        
        # Test with labels
        X_aug, y_aug = augmentor(sequence_data, label_data)
        assert X_aug.shape == sequence_data.shape
        np.testing.assert_array_equal(y_aug, label_data)

# Test PyTorch modules
class TestPyTorchModules:
    """Test PyTorch-compatible modules."""
    
    @pytest.fixture
    def torch_sequence(self, sequence_data):
        """Build a torch tensor fixture for sequence augmentor tests."""
        return torch.from_numpy(sequence_data).float()
    
    def test_random_rc(self, torch_sequence):
        """Test RandomRC module."""
        rc_layer = RandomRC(rc_prob=1.0)  # Always apply RC
        x_aug = rc_layer(torch_sequence)
        
        assert isinstance(x_aug, torch.Tensor)
        assert x_aug.shape == torch_sequence.shape
        
    def test_random_jitter(self, torch_sequence):
        """Test RandomJitter module."""
        jitter = RandomJitter(max_jitter=2)
        x_aug = jitter(torch_sequence)
        
        assert isinstance(x_aug, torch.Tensor)
        assert x_aug.shape[:-1] == torch_sequence.shape[:-1]
        assert x_aug.shape[-1] == torch_sequence.shape[-1] - 4

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
