"""
Behavioral Test Suite for Music Generation Models

This package contains task implementations for evaluating co-creative music models.

Available tasks:
- T1: Structure-Aware Continuation (tasks.t1_structure)
- T2: Style Adherence / Conditioning (tasks.t2_style)
- T3: Edit-Responsiveness / Constraint Satisfaction (tasks.t3_edit)
"""

# Import task executors for convenient access
from .t1_structure import StructureTaskExecutor as T1Executor
from .t2_style import StyleTaskExecutor as T2Executor
from .t3_edit import EditTaskExecutor as T3Executor

__all__ = [
    'T1Executor',
    'T2Executor',
    'T3Executor',
]

__version__ = '0.1.0'
