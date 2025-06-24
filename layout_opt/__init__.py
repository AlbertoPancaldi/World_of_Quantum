"""
Quantum Layout Optimizer for IBM Heavy-Hex Backends

A collection of layout optimization passes designed to outperform
stock Qiskit transpilation on 15-100 qubit circuits for IBM heavy-hex backends.
"""

from .heavyhex_layout import GreedyCommunityLayout
from .clustering import compare_clustering_algorithms
from .distance import HeavyHexTopologyAnalyzer

__version__ = "0.1.0"
__all__ = ['GreedyCommunityLayout', 'compare_clustering_algorithms', 'HeavyHexTopologyAnalyzer'] 