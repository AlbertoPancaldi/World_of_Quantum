"""
Application Circuit Benchmark Suite

Real-world quantum algorithm implementations including QAOA, VQE, 
quantum chemistry, and other application-specific circuits.
"""

from typing import Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, TwoLocal
from .base_benchmark import BaseBenchmark
from config_loader import get_config


class ApplicationBenchmark(BaseBenchmark):
    """
    Benchmark suite with real-world quantum application circuits.
    
    Includes:
    - QAOA (Quantum Approximate Optimization Algorithm)
    - VQE (Variational Quantum Eigensolver) ansätze
    - Quantum Fourier Transform
    - Linear depth circuits
    """
    
    def __init__(self, seed: int = None):
        """Initialize application benchmark suite."""
        config = get_config()
        super().__init__(seed or config.get_seed())
        np.random.seed(self.seed)
    
    def generate_circuits(self, qubit_range: List[int]) -> Dict[str, QuantumCircuit]:
        """
        Generate application circuits for specified qubit counts.
        
        Args:
            qubit_range: List of qubit counts to generate circuits for
            
        Returns:
            Dictionary of application circuits
        """
        circuits = {}
        
        for n_qubits in qubit_range:
            # Generate different types of application circuits
            circuits.update(self._generate_qaoa_circuits(n_qubits))
            circuits.update(self._generate_vqe_circuits(n_qubits))
            circuits.update(self._generate_qft_circuits(n_qubits))
            circuits.update(self._generate_linear_depth_circuits(n_qubits))
        
        print(f"✅ Generated {len(circuits)} application circuits")
        self.cache_circuits(circuits)
        return circuits
    
    def _generate_qaoa_circuits(self, n_qubits: int) -> Dict[str, QuantumCircuit]:
        """Generate QAOA-style circuits."""
        circuits = {}
        config = get_config()
        qaoa_layers = config.get_qaoa_layers()
        
        # QAOA with configurable layers
        qaoa_circuit = QuantumCircuit(n_qubits)
        
        # Initial state: |+⟩^⊗n
        qaoa_circuit.h(range(n_qubits))
        
        # QAOA layers
        for layer in range(qaoa_layers):
            # Cost Hamiltonian (ZZ interactions between adjacent qubits)
            for i in range(n_qubits - 1):
                qaoa_circuit.rzz(np.random.uniform(0, np.pi), i, i + 1)
            
            # Mixer Hamiltonian (X rotations)
            for i in range(n_qubits):
                qaoa_circuit.rx(np.random.uniform(0, np.pi), i)
        
        circuits[f"qaoa_p{qaoa_layers}_{n_qubits}q"] = qaoa_circuit
        return circuits
    
    def _generate_vqe_circuits(self, n_qubits: int) -> Dict[str, QuantumCircuit]:
        """Generate VQE ansatz circuits."""
        circuits = {}
        config = get_config()
        vqe_reps = config.get_vqe_reps()
        
        # Hardware-efficient ansatz
        vqe_circuit = TwoLocal(
            n_qubits, 
            rotation_blocks='ry',
            entanglement_blocks='cz',
            entanglement='linear',
            reps=vqe_reps
        )
        
        # Bind random parameters
        params = np.random.uniform(0, 2*np.pi, vqe_circuit.num_parameters)
        vqe_bound = vqe_circuit.bind_parameters(params)
        
        circuits[f"vqe_twoloc_{n_qubits}q"] = vqe_bound
        return circuits
    
    def _generate_qft_circuits(self, n_qubits: int) -> Dict[str, QuantumCircuit]:
        """Generate Quantum Fourier Transform circuits."""
        circuits = {}
        
        # Standard QFT
        qft_circuit = QFT(n_qubits, approximation_degree=0)
        circuits[f"qft_{n_qubits}q"] = qft_circuit
        
        # Approximate QFT (linear depth)
        if n_qubits >= 10:  # Only for larger circuits
            approx_qft = QFT(n_qubits, approximation_degree=n_qubits//3)
            circuits[f"qft_approx_{n_qubits}q"] = approx_qft
        
        return circuits
    
    def _generate_linear_depth_circuits(self, n_qubits: int) -> Dict[str, QuantumCircuit]:
        """Generate linear depth circuits for comparison."""
        circuits = {}
        
        # Linear depth circuit with nearest-neighbor interactions
        linear_circuit = QuantumCircuit(n_qubits)
        
        # Layer of H gates
        linear_circuit.h(range(n_qubits))
        
        # Linear depth entangling layers
        for layer in range(3):  # 3 layers of linear interactions
            for i in range(n_qubits - 1):
                linear_circuit.cz(i, i + 1)
            for i in range(n_qubits):
                linear_circuit.rz(np.random.uniform(0, 2*np.pi), i)
        
        circuits[f"linear_depth_{n_qubits}q"] = linear_circuit
        return circuits
    
    def get_name(self) -> str:
        """Return benchmark name."""
        return "Application Circuits"
    
    def get_description(self) -> str:
        """Return benchmark description."""
        return (
            "Real-world quantum application circuits including QAOA, VQE ansätze, "
            "Quantum Fourier Transform, and linear depth circuits. Tests layout "
            "optimization on practical quantum algorithms."
        )
    
    def get_circuit_types(self) -> List[str]:
        """Get list of available circuit types."""
        return ["QAOA", "VQE", "QFT", "QFT_Approximate", "Linear_Depth"]


if __name__ == "__main__":
    # Test application benchmark
    from config_loader import load_config
    load_config()  # Load config first
    
    benchmark = ApplicationBenchmark()  # Uses config values
    
    # Generate test circuits
    test_sizes = [15, 25]
    circuits = benchmark.generate_circuits(test_sizes)
    
    print(f"\n🔬 Application Benchmark Test:")
    print(f"  Circuit types: {benchmark.get_circuit_types()}")
    print(f"  Generated circuits:")
    
    for name, circuit in circuits.items():
        info = benchmark.get_circuit_info(name)
        print(f"    {name}: {info['num_qubits']} qubits, depth {info['depth']}")
        print(f"      Gates: {info['gate_counts']}")
