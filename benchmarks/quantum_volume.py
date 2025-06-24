"""
Quantum Volume Benchmark Suite

Implements the industry-standard Quantum Volume benchmark circuits
for testing quantum layout optimization algorithms.

Quantum Volume circuits are square (width = depth) random circuits
that stress-test both connectivity and gate fidelity.
"""

from typing import Dict, List
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from .base_benchmark import BaseBenchmark
from config_loader import get_config


class QuantumVolumeBenchmark(BaseBenchmark):
    """
    Quantum Volume benchmark implementation.
    
    Generates standardized QV circuits with configurable sizes.
    QV circuits are ideal for layout optimization testing due to
    their heavy use of 2-qubit gates and random structure.
    """
    
    def __init__(self, seed: int = None, depth_factor: float = None):
        """
        Initialize Quantum Volume benchmark.
        
        Args:
            seed: Random seed for reproducible circuits (uses config if None)
            depth_factor: Multiplier for circuit depth (uses config if None)
        """
        config = get_config()
        super().__init__(seed or config.get_seed())
        self.depth_factor = depth_factor or config.get_depth_factor()
    
    def generate_circuits(self, qubit_range: List[int]) -> Dict[str, QuantumCircuit]:
        """
        Generate Quantum Volume circuits for specified qubit counts.
        
        Args:
            qubit_range: List of qubit counts (e.g., [15, 25, 50, 75, 100])
            
        Returns:
            Dictionary mapping circuit names to QV circuits
        """
        circuits = {}
        
        for n_qubits in qubit_range:
            # Calculate circuit depth (typically equal to width for QV)
            depth = max(1, int(n_qubits * self.depth_factor))
            
            # Generate QV circuit
            circuit_name = f"qv_{n_qubits}"
            circuits[circuit_name] = QuantumVolume(
                n_qubits, 
                depth=depth, 
                seed=self.seed
            )
            
            print(f"Generated {circuit_name}: {n_qubits} qubits, depth {depth}")
        
        # Cache the generated circuits
        self.cache_circuits(circuits)
        return circuits
    
    def get_name(self) -> str:
        """Return benchmark name."""
        return "Quantum Volume"
    
    def get_description(self) -> str:
        """Return benchmark description."""
        return (
            "Industry-standard Quantum Volume benchmark circuits. "
            "Square random circuits with heavy 2-qubit gate usage, "
            "ideal for testing layout optimization effectiveness."
        )
    
    def get_expected_qv_level(self, n_qubits: int) -> int:
        """
        Get the expected Quantum Volume level for a given circuit size.
        
        Args:
            n_qubits: Number of qubits in the circuit
            
        Returns:
            Expected QV level (2^n_qubits for perfect execution)
        """
        return 2 ** n_qubits
    
    def generate_size_series(self, max_qubits: int = 100, step: int = 10) -> List[int]:
        """
        Generate a series of qubit counts for benchmarking.
        
        Args:
            max_qubits: Maximum number of qubits
            step: Step size between circuit sizes
            
        Returns:
            List of qubit counts suitable for benchmarking
        """
        # Start with some standard small sizes
        sizes = [15, 20, 25, 30]
        
        # Add larger sizes with specified step
        current = 40
        while current <= max_qubits:
            sizes.append(current)
            current += step
        
        return sorted(list(set(sizes)))  # Remove duplicates and sort


if __name__ == "__main__":
    # Test the Quantum Volume benchmark
    from config_loader import load_config
    load_config()  # Load config first
    
    benchmark = QuantumVolumeBenchmark()  # Uses config values
    
    # Generate test circuits
    test_sizes = [15, 25, 50]
    circuits = benchmark.generate_circuits(test_sizes)
    
    print(f"\n✅ Generated {len(circuits)} Quantum Volume circuits:")
    for name, circuit in circuits.items():
        info = benchmark.get_circuit_info(name)
        print(f"  {name}: {info['num_qubits']} qubits, depth {info['depth']}")
        
    print(f"\n🎯 Expected QV levels:")
    for size in test_sizes:
        qv_level = benchmark.get_expected_qv_level(size)
        print(f"  {size} qubits → QV {qv_level}")
