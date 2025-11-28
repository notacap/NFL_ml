"""
NFL Data Loaders Module

Provides modular data loading infrastructure for building NFL prediction datasets.

Architecture:
- BaseDataLoader: Abstract base class for all loaders
- Individual loaders (in loaders/): Handle specific data sources
- NFLDatasetBuilder: Orchestrates loading and joining (in build_dataset.py)

Usage:
    from src.data_loaders import BaseDataLoader
    from src.data_loaders.loaders import PlayerReceivingLoader  # when implemented

Author: Claude Code
Created: 2024-11-27
"""

from .base_loader import BaseDataLoader

__all__ = ['BaseDataLoader']
