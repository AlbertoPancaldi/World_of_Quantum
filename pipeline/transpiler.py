"""
Quantum Circuit Transpilation Comparison

This module provides tools for comparing stock Qiskit transpilation 
with custom layout optimization approaches. 

Now uses comprehensive metrics from layout_opt.utils for consistency
and to avoid code duplication.
"""

import time
from typing import Dict, List, Any, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend

from config_loader import get_config
# Import comprehensive metrics instead of duplicating logic
from layout_opt.utils import transpile_and_score, compute_comprehensive_comparison


class TranspilerComparison:
    """
    Compare transpilation results between stock Qiskit and custom layout passes.
    
    Now uses the comprehensive metrics system from layout_opt.utils to ensure
    consistent, detailed performance analysis including error-weighted metrics.
    """
    
    def __init__(self, backend: Backend, optimization_level: int = None):
        """Initialize transpiler comparison."""
        self.backend = backend
        config = get_config()
        self.optimization_level = optimization_level or config.get_optimization_level()
        
    def compare_transpilation(self, 
                            circuit: QuantumCircuit,
                            custom_layout_pass: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compare stock vs custom transpilation with comprehensive metrics.
        
        Args:
            circuit: Quantum circuit to transpile and compare
            custom_layout_pass: Custom layout pass to test (None for stock only)
            
        Returns:
            Comprehensive comparison results with detailed metrics
        """
        print(f"🔄 Running stock transpilation (opt_level={self.optimization_level})...")
        
        # Get input circuit statistics for context
        input_stats = self._get_input_stats(circuit)
        
        # Run stock transpilation using comprehensive metrics
        stock_metrics = transpile_and_score(
            circuit, 
            self.backend, 
            layout_pass=None,  # Stock transpilation
            optimization_level=self.optimization_level
        )
        
        # Prepare stock result in expected format
        stock_result = {
            'success': stock_metrics.get('success', True),
            'compile_time': stock_metrics.get('compile_time', 0.0),
            'stats': self._convert_metrics_to_stats(stock_metrics),
            'error_message': stock_metrics.get('error_message')
        }
        
        # Run custom transpilation if layout pass provided
        custom_result = None
        if custom_layout_pass is not None:
            print(f"🔄 Running custom transpilation...")
            
            custom_metrics = transpile_and_score(
                circuit, 
                self.backend, 
                layout_pass=custom_layout_pass,
                optimization_level=self.optimization_level
            )
            
            custom_result = {
                'success': custom_metrics.get('success', True),
                'compile_time': custom_metrics.get('compile_time', 0.0),
                'stats': self._convert_metrics_to_stats(custom_metrics),
                'error_message': custom_metrics.get('error_message')
            }
        
        # Compute comprehensive comparison using our improved metrics
        comparison = None
        if custom_result and stock_result['success'] and custom_result['success']:
            comparison = compute_comprehensive_comparison(
                custom_metrics, stock_metrics
            )
        elif custom_result:
            comparison = {'comparison_valid': False}
        
        return {
            'circuit_name': getattr(circuit, 'name', f'circuit_{circuit.num_qubits}q'),
            'input_stats': input_stats,
            'stock': stock_result,
            'custom': custom_result,
            'comparison': comparison
        }
    
    def _get_input_stats(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Get basic statistics about the input circuit."""
        # Check if circuit contains high-level gates that need decomposition
        gate_counts = circuit.count_ops()
        has_high_level_gates = any('quantum_volume' in gate_name or 
                                 gate_name in ['qft', 'iqft', 'qaoa', 'vqe'] 
                                 for gate_name in gate_counts.keys())
        
        if has_high_level_gates:
            # Decompose multiple levels to get actual gate counts
            decomposed_circuit = circuit.decompose()
            
            # Keep decomposing until we reach basic gates or no change occurs
            for level in range(1, 5):  # Max 5 levels to avoid infinite loops
                prev_gates = set(decomposed_circuit.count_ops().keys())
                next_decomp = decomposed_circuit.decompose()
                next_gates = set(next_decomp.count_ops().keys())
                
                # Stop if no new gate types appear (fully decomposed)
                if prev_gates == next_gates:
                    break
                    
                decomposed_circuit = next_decomp
                
                # Stop if we have basic 2-qubit gates
                if any(gate in next_gates for gate in ['cx', 'cz', 'ecr']):
                    break
            
            gate_counts = decomposed_circuit.count_ops()
            circuit_depth = decomposed_circuit.depth()
            print(f"📊 Decomposed input circuit ({level} levels): {len(gate_counts)} gate types, depth {circuit_depth}")
            print(f"    Gate breakdown: {dict(gate_counts)}")
        else:
            circuit_depth = circuit.depth()
        
        two_qubit_gates = {'cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 'iswap', 'swap'}
        cx_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
        
        return {
            'num_qubits': circuit.num_qubits,
            'depth': circuit_depth,
            'cx_count': cx_count,
            'total_gates': sum(gate_counts.values())
        }
    
    def _convert_metrics_to_stats(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert comprehensive metrics to the stats format expected by the notebook.
        
        This maintains backward compatibility while using comprehensive metrics.
        """
        if not metrics.get('success', True):
            return {}
            
        return {
            'num_qubits': metrics.get('n_qubits', 0),
            'depth': metrics.get('depth', 0),
            'gate_counts': metrics.get('gate_counts', {}),
            'cx_count': metrics.get('cx_count', 0),
            'single_qubit_count': metrics.get('single_qubit_count', 0),
            'total_gates': metrics.get('total_gates', 0),
            
            # New comprehensive metrics (bonus!)
            'error_weighted_cx': metrics.get('error_weighted_cx', 0.0),
            'error_weighted_total': metrics.get('error_weighted_total', 0.0),
            'connectivity_score': metrics.get('connectivity_score', 0.0),
            'gate_density': metrics.get('gate_density', 0.0),
            'depth_efficiency': metrics.get('depth_efficiency', 0.0),
            'ecr_count': metrics.get('ecr_count', 0),
            'sx_count': metrics.get('sx_count', 0),
            'rz_count': metrics.get('rz_count', 0)
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
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
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
        print(f"  Error-weighted CX: {result['stock']['stats']['error_weighted_cx']:.4f}")
        print(f"  Connectivity score: {result['stock']['stats']['connectivity_score']:.3f}")
