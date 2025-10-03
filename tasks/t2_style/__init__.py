"""
T2: Style Adherence / Conditioning Task Module

This module provides functionality for testing style consistency in music generation.
"""
from .style_task import (
    StyleTaskExecutor,
    TaskInput,
    TaskResult,
    StyleSpec,
    StyleCategory,
    ModelType
)

__all__ = [
    'StyleTaskExecutor',
    'TaskInput',
    'TaskResult',
    'StyleSpec',
    'StyleCategory',
    'ModelType'
]
