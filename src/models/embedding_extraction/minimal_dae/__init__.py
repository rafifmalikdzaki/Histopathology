"""
Minimal DAE Model for Embedding Extraction.

This module provides a simplified DAE model focused only on the encoder and bottleneck
components needed for embedding extraction.
"""

from .model import MinimalDAEModel, MinimalBottleneck, load_minimal_model
from .extractor import MinimalDAEExtractor

__all__ = [
    'MinimalDAEModel',
    'MinimalBottleneck',
    'load_minimal_model',
    'MinimalDAEExtractor',
]
