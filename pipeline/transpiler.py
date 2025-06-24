"""
Transpiler Comparison Engine

Orchestrates comparison between custom layout optimization passes
and stock Qiskit transpilation. Handles PassManager construction,
transpilation execution, and results collection.
"""

from typing import Dict, List, Optional, Any, Tuple
import time
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.providers import Backend
from config_loader import get_config


class TranspilerComparison:
    """
    Manages comparison between custom and stock transpilation.
    
    Provides a unified interface for running both custom layout optimization
    and stock Qiskit transpilation, collecting metrics from both approaches.
    """
    
    def __init__(self, backend: Backend, optimization_level: int = None):
        """
        Initialize transpiler comparison.
        
        Args:
            backend: Target quantum backend (IBM Brisbane)
            optimization_level: Qiskit optimization level for baseline (uses config if None)
        """
        config = get_config()
        
        self.backend = backend
        self.optimization_level = optimization_level or config.get_optimization_level()
        self.coupling_map = backend.coupling_map
        
        # Cache for expensive computations
        self._distance_cache = {}
        
    def compare_transpilation(self, 
                            circuit: QuantumCircuit,
                            custom_layout_pass: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run both custom and stock transpilation and compare results.
        
        Args:
            circuit: Input quantum circuit
            custom_layout_pass: Custom layout pass (None for stock only)
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            'circuit_name': getattr(circuit, 'name', 'unnamed'),
            'input_stats': self._get_circuit_stats(circuit)
        }
        
        # Run stock transpilation
        print(f"🔄 Running stock transpilation (opt_level={self.optimization_level})...")
        stock_result = self._transpile_stock(circuit)
        results['stock'] = stock_result
        
        # Run custom transpilation if layout pass provided
        if custom_layout_pass is not None:
            print(f"🔄 Running custom transpilation...")
            custom_result = self._transpile_custom(circuit, custom_layout_pass)
            results['custom'] = custom_result
            
            # Compute comparison metrics
            results['comparison'] = self._compute_comparison_metrics(
                custom_result, stock_result
            )
        else:
            print("⚠️  No custom layout pass provided, skipping custom transpilation")
            results['custom'] = None
            results['comparison'] = None
        
        return results
    
    def _transpile_stock(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Run stock Qiskit transpilation."""
        start_time = time.time()
        
        try:
            # Use stock Qiskit preset pass manager
            config = get_config()
            transpiled = transpile(
                circuit, 
                backend=self.backend, 
                optimization_level=self.optimization_level,
                seed_transpiler=config.get_seed()  # For reproducibility
            )
            
            compile_time = time.time() - start_time
            
            return {
                'transpiled_circuit': transpiled,
                'success': True,
                'compile_time': compile_time,
                'stats': self._get_circuit_stats(transpiled),
                'layout': getattr(transpiled, '_layout', None),
                'error_message': None
            }
            
        except Exception as e:
            return {
                'transpiled_circuit': None,
                'success': False,
                'compile_time': time.time() - start_time,
                'stats': {},
                'layout': None,
                'error_message': str(e)
            }
    
    def _transpile_custom(self, 
                         circuit: QuantumCircuit, 
                         layout_pass: Any) -> Dict[str, Any]:
        """Run custom transpilation with layout pass."""
        start_time = time.time()
        
        try:
            config = get_config()
            
            # APPROACH 1: Use transpile() with initial_layout (RECOMMENDED)
            # First, run our custom layout pass to get the layout
            layout_pm = PassManager([layout_pass])
            temp_circuit = layout_pm.run(circuit.copy())
            
            # Extract the layout from our custom pass
            if hasattr(temp_circuit, '_layout') and temp_circuit._layout:
                # Get physical qubit mapping from our custom layout
                initial_layout = temp_circuit._layout.get_physical_bits()
                
                # Use standard Qiskit transpile with our custom initial layout
                # This ensures all other stages (routing, optimization, scheduling) are handled properly
                transpiled = transpile(
                    circuit,
                    backend=self.backend,
                    optimization_level=self.optimization_level,  # Match stock optimization level
                    initial_layout=initial_layout,
                    seed_transpiler=config.get_seed()
                )
                
            else:
                # Fallback: if layout pass doesn't set _layout properly, 
                # try to extract layout differently or use standard transpile
                print("⚠️  Custom layout pass didn't set _layout, using standard transpile")
                transpiled = transpile(
                    circuit,
                    backend=self.backend,
                    optimization_level=self.optimization_level,
                    seed_transpiler=config.get_seed()
                )
            
            compile_time = time.time() - start_time
            
            return {
                'transpiled_circuit': transpiled,
                'success': True,
                'compile_time': compile_time,
                'stats': self._get_circuit_stats(transpiled),
                'layout': getattr(transpiled, '_layout', None),
                'error_message': None
            }
            
        except Exception as e:
            return {
                'transpiled_circuit': None,
                'success': False,
                'compile_time': time.time() - start_time,
                'stats': {},
                'layout': None,
                'error_message': str(e)
            }
    
    def _get_circuit_stats(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Extract comprehensive circuit statistics."""
        gate_counts = circuit.count_ops()
        
        # Count 2-qubit gates specifically
        two_qubit_gates = ['cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx']
        cx_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
        
        # Count single-qubit gates
        single_qubit_count = sum(count for gate, count in gate_counts.items() 
                               if gate not in two_qubit_gates)
        
        return {
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'gate_counts': gate_counts,
            'cx_count': cx_count,
            'single_qubit_count': single_qubit_count,
            'total_gates': sum(gate_counts.values())
        }
    
    def _compute_comparison_metrics(self, 
                                  custom_result: Dict[str, Any],
                                  stock_result: Dict[str, Any]) -> Dict[str, float]:
        """Compute comparison metrics between custom and stock results."""
        if not (custom_result['success'] and stock_result['success']):
            return {'comparison_valid': False}
        
        custom_stats = custom_result['stats']
        stock_stats = stock_result['stats']
        
        # Compute reduction percentages
        def compute_reduction(custom_val, stock_val):
            if stock_val == 0:
                return 0.0
            return (1.0 - custom_val / stock_val) * 100.0
        
        return {
            'comparison_valid': True,
            'cx_reduction_percent': compute_reduction(
                custom_stats['cx_count'], stock_stats['cx_count']
            ),
            'depth_reduction_percent': compute_reduction(
                custom_stats['depth'], stock_stats['depth']
            ),
            'total_gates_reduction_percent': compute_reduction(
                custom_stats['total_gates'], stock_stats['total_gates']
            ),
            'compile_time_ratio': (
                custom_result['compile_time'] / stock_result['compile_time']
                if stock_result['compile_time'] > 0 else float('inf')
            ),
            'absolute_cx_reduction': (
                stock_stats['cx_count'] - custom_stats['cx_count']
            ),
            'absolute_depth_reduction': (
                stock_stats['depth'] - custom_stats['depth']
            )
        }
    
    def batch_compare(self, 
                     circuits: Dict[str, QuantumCircuit],
                     custom_layout_pass: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Run comparison on a batch of circuits.
        
        Args:
            circuits: Dictionary of circuits to compare
            custom_layout_pass: Custom layout pass
            
        Returns:
            List of comparison results
        """
        results = []
        
        print(f"🚀 Starting batch comparison of {len(circuits)} circuits...")
        
        for i, (name, circuit) in enumerate(circuits.items()):
            print(f"\n[{i+1}/{len(circuits)}] Processing {name}...")
            
            # Set circuit name for tracking
            circuit.name = name
            
            # Run comparison
            result = self.compare_transpilation(circuit, custom_layout_pass)
            results.append(result)
            
            # Print summary
            if result['comparison'] and result['comparison']['comparison_valid']:
                cx_reduction = result['comparison']['cx_reduction_percent']
                print(f"  ✅ CX reduction: {cx_reduction:.1f}%")
            else:
                print(f"  ⚠️  Comparison not available")
        
        print(f"\n🎉 Batch comparison complete!")
        return results


if __name__ == "__main__":
    # Test transpiler comparison
    from qiskit.providers.fake_provider import FakeBrisbane
    from qiskit.circuit.library import QuantumVolume
    
    backend = FakeBrisbane()
    comparator = TranspilerComparison(backend)
    
    # Load config and test with simple circuit
    from config_loader import load_config
    config = load_config()
    
    test_circuit = QuantumVolume(15, depth=5, seed=config.get_seed())
    test_circuit.name = "test_qv_15"
    
    result = comparator.compare_transpilation(test_circuit)
    
    print(f"\n🧪 Transpiler Comparison Test:")
    print(f"  Circuit: {result['circuit_name']}")
    print(f"  Stock success: {result['stock']['success']}")
    if result['stock']['success']:
        print(f"  Stock CX count: {result['stock']['stats']['cx_count']}")
        print(f"  Stock depth: {result['stock']['stats']['depth']}")
