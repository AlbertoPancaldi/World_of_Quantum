"""
Abstract Base Class for Quantum Circuit Benchmarks

Defines the interface that all benchmark suites must implement.
Provides common functionality for circuit generation and management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from qiskit import QuantumCircuit
from config_loader import get_config


class BaseBenchmark(ABC):
    """
    Abstract base class for all quantum circuit benchmark suites.
    
    All benchmark implementations must inherit from this class and
    implement the required abstract methods.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize benchmark suite.
        
        Args:
            seed: Random seed for reproducible circuit generation (uses config if None)
        """
        config = get_config()
        self.seed = seed or config.get_seed()
        self._circuits_cache = {}
    
    @abstractmethod
    def generate_circuits(self, qubit_range: List[int]) -> Dict[str, QuantumCircuit]:
        """
        Generate benchmark circuits for specified qubit counts.
        
        Args:
            qubit_range: List of qubit counts to generate circuits for
            
        Returns:
            Dictionary mapping circuit names to QuantumCircuit objects
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of this benchmark suite.
        
        Returns:
            Human-readable benchmark suite name
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Return a description of this benchmark suite.
        
        Returns:
            Detailed description of the benchmark suite
        """
        pass
    
    def get_circuit_info(self, circuit_name: str) -> Dict[str, Any]:
        """
        Get information about a specific circuit.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            Dictionary with circuit metadata
        """
        if circuit_name in self._circuits_cache:
            circuit = self._circuits_cache[circuit_name]
            return {
                'name': circuit_name,
                'num_qubits': circuit.num_qubits,
                'depth': circuit.depth(),
                'gate_counts': circuit.count_ops(),
                'benchmark_suite': self.get_name()
            }
        return {}
    
    def cache_circuits(self, circuits: Dict[str, QuantumCircuit]) -> None:
        """
        Cache generated circuits for future use.
        
        Args:
            circuits: Dictionary of circuits to cache
        """
        self._circuits_cache.update(circuits)
    
    def get_cached_circuits(self) -> Dict[str, QuantumCircuit]:
        """
        Get all cached circuits.
        
        Returns:
            Dictionary of cached circuits
        """
        return self._circuits_cache.copy()
    
    def clear_cache(self) -> None:
        """Clear the circuit cache."""
        self._circuits_cache.clear()
