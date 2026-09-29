"""Pytest fixtures for the fundamental-group verification suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Return an independent deterministic generator for each test."""
    return np.random.default_rng(0)
