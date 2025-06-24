"""
QASM Circuit Benchmark Suite

Loads and manages quantum circuits from QASM files.
Useful for testing with real-world circuits and external benchmarks.
"""

from typing import Dict, List, Optional
from pathlib import Path
from qiskit import QuantumCircuit, qasm2
from .base_benchmark import BaseBenchmark
from config_loader import get_config


class QASMBenchmark(BaseBenchmark):
    """
    Benchmark suite for circuits loaded from QASM files.
    
    Supports loading circuits from a directory of QASM files
    and filtering by qubit count ranges.
    """
    
    def __init__(self, qasm_directory: str = None, seed: int = None):
        """
        Initialize QASM benchmark suite.
        
        Args:
            qasm_directory: Directory containing QASM files (uses config if None)
            seed: Random seed (uses config if None)
        """
        config = get_config()
        super().__init__(seed or config.get_seed())
        
        if qasm_directory is None:
            qasm_directory = config.raw_config.get('benchmarks', {}).get('qasm_circuits', {}).get('directory', 'benchmarks/circuits')
        
        self.qasm_directory = Path(qasm_directory)
        
    def generate_circuits(self, qubit_range: List[int]) -> Dict[str, QuantumCircuit]:
        """
        Load QASM circuits that fall within the specified qubit range.
        
        Args:
            qubit_range: List of acceptable qubit counts
            
        Returns:
            Dictionary of loaded circuits within qubit range
        """
        circuits = {}
        
        if not self.qasm_directory.exists():
            print(f"⚠️  QASM directory {self.qasm_directory} not found")
            return circuits
        
        # Find all QASM files
        qasm_files = list(self.qasm_directory.glob("*.qasm"))
        if not qasm_files:
            print(f"⚠️  No QASM files found in {self.qasm_directory}")
            return circuits
        
        min_qubits, max_qubits = min(qubit_range), max(qubit_range)
        
        for qasm_file in qasm_files:
            try:
                # Load circuit from QASM
                circuit = qasm2.load(str(qasm_file))
                
                # Check if circuit size is in range
                if min_qubits <= circuit.num_qubits <= max_qubits:
                    circuit_name = f"qasm_{qasm_file.stem}_{circuit.num_qubits}q"
                    circuits[circuit_name] = circuit
                    print(f"Loaded {circuit_name}: {circuit.num_qubits} qubits, depth {circuit.depth()}")
                    
            except Exception as e:
                print(f"⚠️  Failed to load {qasm_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(circuits)} QASM circuits")
        self.cache_circuits(circuits)
        return circuits
    
    def get_name(self) -> str:
        """Return benchmark name."""
        return "QASM Circuits"
    
    def get_description(self) -> str:
        """Return benchmark description."""
        return (
            "Benchmark suite using quantum circuits loaded from QASM files. "
            "Useful for testing with real-world circuits and external benchmarks."
        )
    
    def add_qasm_file(self, qasm_path: str, circuit_name: Optional[str] = None) -> bool:
        """
        Add a single QASM file to the benchmark suite.
        
        Args:
            qasm_path: Path to QASM file
            circuit_name: Optional custom name for the circuit
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            circuit = qasm2.load(qasm_path)
            name = circuit_name or f"qasm_{Path(qasm_path).stem}_{circuit.num_qubits}q"
            self._circuits_cache[name] = circuit
            return True
        except Exception as e:
            print(f"⚠️  Failed to load {qasm_path}: {e}")
            return False


if __name__ == "__main__":
    # Test QASM benchmark
    benchmark = QASMBenchmark()
    
    # Try to load circuits (will show warning if directory doesn't exist)
    test_range = [10, 20, 30, 50]
    circuits = benchmark.generate_circuits(test_range)
    
    print(f"\n📁 QASM Benchmark Test:")
    print(f"  Directory: {benchmark.qasm_directory}")
    print(f"  Circuits found: {len(circuits)}")
