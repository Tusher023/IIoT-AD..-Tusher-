"""Tests for reproducibility utilities."""

import numpy as np
import torch
from src.utils.reproducibility import set_seed, get_device


def test_set_seed_deterministic():
    """Verify that setting the same seed produces identical random numbers."""
    set_seed(42)
    a1 = np.random.rand(5)
    t1 = torch.rand(5)
    
    set_seed(42)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)
    
    np.testing.assert_array_equal(a1, a2)
    assert torch.equal(t1, t2)


def test_get_device():
    """Verify that get_device returns a valid torch device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ('cpu', 'cuda')
