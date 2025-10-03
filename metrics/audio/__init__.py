"""
Audio metrics module for music generation evaluation.

This module provides comprehensive audio quality and music-specific metrics
for evaluating generated music, including:
- FAD (Fréchet Audio Distance)
- CLAP Score (text-audio alignment)
- Tempo/Beat consistency
- Key stability
- Structure detection
"""

from .fad import FADCalculator
from .clapscore import CLAPScoreCalculator
from .tempo import TempoConsistencyCalculator
from .key_stability import KeyStabilityCalculator
from .structure import StructureDetectionCalculator

__all__ = [
    "FADCalculator",
    "CLAPScoreCalculator",
    "TempoConsistencyCalculator",
    "KeyStabilityCalculator",
    "StructureDetectionCalculator",
]
