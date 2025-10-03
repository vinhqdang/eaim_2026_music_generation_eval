"""
T1: Structure-Aware Continuation Task Module

This module provides functionality for testing structure-aware music generation.
"""
from .structure_task import (
    StructureTaskExecutor,
    TaskInput,
    TaskResult,
    StructureSpec,
    ModelType,
    EditCompliance
)

__all__ = [
    'StructureTaskExecutor',
    'TaskInput',
    'TaskResult',
    'StructureSpec',
    'ModelType',
    'EditCompliance'
]
