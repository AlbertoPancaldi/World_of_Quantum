"""
Transpilation Pipeline for Quantum Layout Optimization

Orchestrates the comparison between custom layout optimization
and stock Qiskit transpilation across different benchmarks.

Key components:
- TranspilerComparison: Manages custom vs stock transpilation
- MetricsCollector: Collects and analyzes performance metrics
- ResultsManager: Handles results storage and visualization
"""

from .transpiler import TranspilerComparison
from .metrics import MetricsCollector
from .results import ResultsManager

__all__ = [
    'TranspilerComparison',
    'MetricsCollector', 
    'ResultsManager'
]

__version__ = "0.1.0"
