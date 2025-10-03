"""
T3: Edit-Responsiveness (Constraint Satisfaction) Task Module

This module provides functionality for testing edit responsiveness in music generation.
"""
from .edit_task import (
    EditTaskExecutor,
    TaskInput,
    TaskResult,
    Edit,
    EditType,
    EditCompliance,
    ModelType
)

__all__ = [
    'EditTaskExecutor',
    'TaskInput',
    'TaskResult',
    'Edit',
    'EditType',
    'EditCompliance',
    'ModelType'
]
