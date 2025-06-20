"""
Training Module for DAE-KAN-Attention Model

This module provides training utilities and implementations for the DAE-KAN-Attention model,
including robust training with curriculum learning, advanced logging, and visualization.
"""

from .dae_kan_attention.pl_training_robust import (
    RobustDAE,
    train_robust,
    AdvancedWandbCallback
)

__all__ = [
    "RobustDAE",
    "train_robust",
    "AdvancedWandbCallback",
]
