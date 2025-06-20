"""
Histopathology Image Analysis with DAE-KAN-Attention Model
"""

__version__ = "1.0.0"

from .models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention
from .training.dae_kan_attention.pl_training_robust import RobustDAE, train_robust

__all__ = [
    "DAE_KAN_Attention",
    "RobustDAE",
    "train_robust",
]
