"""
Utility Functions for Layout Optimization

Benchmarking, metrics computation, and visualization utilities for
evaluating layout optimization performance against stock Qiskit transpilation.

Key functions:
- load_benchmarks(): Load quantum circuits for testing
- transpile_and_score(): Transpile and compute metrics
- plot_metrics(): Visualize optimization results
- convenience functions for interaction graph construction
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.transpiler import PassManager
from qiskit.circuit.library import QuantumVolume
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import dag_to_circuit

from config_loader import get_config
from .circuit_analyzer import CircuitAnalyzer
from .benchmark_runner import BenchmarkRunner


def interaction_graph(circuit: QuantumCircuit) -> nx.Graph:
    """
    Convenience function to build interaction graph from QuantumCircuit.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Logical circuit whose 2-qubit interactions you want to count.

    Returns
    -------
    networkx.Graph
        Simple undirected graph; edge weights = interaction counts.
    """
    analyzer = CircuitAnalyzer()
    return analyzer.build_interaction_graph(circuit)


def interaction_graph_from_dag(dag: DAGCircuit) -> nx.Graph:
    """
    Convenience function to build interaction graph from DAGCircuit.
    
    Args:
        dag: Qiskit DAGCircuit
        
    Returns:
        Weighted interaction graph
    """
    analyzer = CircuitAnalyzer()
    return analyzer.build_interaction_graph_from_dag(dag)


def build_interaction_graph(dag: DAGCircuit) -> nx.Graph:
    """
    Main function used by layout passes to build interaction graph from DAG.
    
    Args:
        dag: Qiskit DAGCircuit
        
    Returns:
        Weighted interaction graph
    """
    return interaction_graph_from_dag(dag)


def load_benchmarks(benchmark_dir: str = "benchmarks/") -> Dict[str, QuantumCircuit]:
    """
    Load benchmark quantum circuits for testing.
    
    Args:
        benchmark_dir: Directory containing QASM files
        
    Returns:
        Dictionary mapping circuit names to QuantumCircuit objects
    """
    # TODO: Load QASM files from benchmark directory
    # TODO: Generate Quantum Volume circuits (15-85 qubits)
    # TODO: Create QAOA and VQE test circuits
    
    circuits = {}
    
    # Placeholder: create simple test circuits using config
    config = get_config()
    for n_qubits in [15, 25, 35, 50, 75]:
        circuits[f"qv_{n_qubits}"] = QuantumVolume(
            n_qubits, 
            depth=int(n_qubits * config.get_depth_factor()), 
            seed=config.get_seed()
        )
    
    return circuits


def transpile_and_score(circuit: QuantumCircuit, 
                       backend: Backend,
                       layout_pass: Optional[Any] = None,
                       optimization_level: int = None) -> Dict[str, float]:
    """
    Transpile circuit and compute comprehensive performance metrics.
    
    Args:
        circuit: Input quantum circuit
        backend: Target quantum backend
        layout_pass: Custom layout pass (None for stock Qiskit)
        optimization_level: Qiskit optimization level
        
    Returns:
        Dictionary of performance metrics including gate counts, errors, and timing
    """
    start_time = time.time()
    config = get_config()
    opt_level = optimization_level or config.get_optimization_level()
    
    try:
        if layout_pass is not None:
            # Use custom layout pass with proper integration
            from qiskit.converters import circuit_to_dag
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            
            print("🔄 Using custom layout pass...")
            
            # Get layout from custom pass
            dag = circuit_to_dag(circuit)
            layout_dag = layout_pass.run(dag)
            layout = layout_pass.property_set.get('layout', None)
            
            if layout is None:
                raise ValueError("Custom layout pass didn't set layout")
            
            # Use preset pass manager with custom initial layout
            transpiled = transpile(
                circuit,
                backend=backend,
                optimization_level=opt_level,
                initial_layout=layout,
                seed_transpiler=config.get_seed(),
                layout_method=None,  # Don't run layout again
                routing_method='sabre'
            )
            
        else:
            # Use stock Qiskit transpilation
            transpiled = transpile(
                circuit, 
                backend=backend, 
                optimization_level=opt_level,
                seed_transpiler=config.get_seed()
            )
        
        compile_time = time.time() - start_time
        
        # Compute comprehensive circuit statistics
        stats = _compute_comprehensive_stats(transpiled, backend)
        stats['compile_time'] = compile_time
        stats['success'] = True
        stats['error_message'] = None
        
        return stats
        
    except Exception as e:
        compile_time = time.time() - start_time
        print(f"❌ Transpilation failed: {str(e)}")
        
        return {
            'success': False,
            'compile_time': compile_time,
            'error_message': str(e),
            'cx_count': 0,
            'single_qubit_count': 0,
            'depth': 0,
            'error_weighted_cx': 0.0,
            'n_qubits': circuit.num_qubits,
            'total_gates': 0
        }


def _compute_comprehensive_stats(circuit: QuantumCircuit, backend: Backend) -> Dict[str, float]:
    """
    Compute comprehensive circuit statistics including error-weighted metrics.
    
    Args:
        circuit: Transpiled quantum circuit
        backend: Target quantum backend for error rates
        
    Returns:
        Dictionary of circuit statistics and metrics
    """
    # Get basic gate counts using Qiskit's count_ops
    gate_counts = circuit.count_ops()
    
    # Define 2-qubit gates (comprehensive list for IBM backends)
    two_qubit_gates = {
        'cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 
        'iswap', 'swap', 'dcx', 'ch', 'crx', 'cry', 'crz',
        'cu', 'cu1', 'cu3', 'ccx', 'csx', 'cswap'
    }
    
    # Count different gate types
    cx_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
    single_qubit_count = sum(count for gate, count in gate_counts.items() 
                           if gate not in two_qubit_gates and gate not in ['measure', 'barrier'])
    
    # Total operational gates (excluding measurements and barriers)
    operational_gates = sum(count for gate, count in gate_counts.items() 
                          if gate not in ['measure', 'barrier'])
    
    # Get circuit layout information for error calculation
    layout = getattr(circuit, '_layout', None)
    
    # Compute error-weighted metrics
    error_weighted_cx = _compute_error_weighted_cx(circuit, backend, layout)
    error_weighted_total = _compute_error_weighted_total(circuit, backend, layout)
    
    # Compute additional depth and connectivity metrics
    connectivity_score = _compute_connectivity_score(circuit, backend)
    
    return {
        # Basic counts
        'n_qubits': circuit.num_qubits,
        'depth': circuit.depth(),
        'cx_count': cx_count,
        'single_qubit_count': single_qubit_count,
        'total_gates': operational_gates,
        'total_ops': sum(gate_counts.values()),  # Including measurements
        
        # Gate type breakdown
        'gate_counts': gate_counts,
        
        # Error-weighted metrics
        'error_weighted_cx': error_weighted_cx,
        'error_weighted_total': error_weighted_total,
        
        # Performance metrics
        'connectivity_score': connectivity_score,
        'gate_density': operational_gates / circuit.num_qubits if circuit.num_qubits > 0 else 0,
        'depth_efficiency': operational_gates / circuit.depth() if circuit.depth() > 0 else 0,
        
        # Specific IBM backend metrics
        'ecr_count': gate_counts.get('ecr', 0),  # IBM's native 2-qubit gate
        'sx_count': gate_counts.get('sx', 0),    # IBM's native single-qubit gate
        'rz_count': gate_counts.get('rz', 0),    # IBM's virtual Z rotation
    }


def _compute_error_weighted_cx(circuit: QuantumCircuit, backend: Backend, 
                              layout: Optional[Any] = None) -> float:
    """
    Compute error-weighted 2-qubit gate count using backend error rates.
    
    Args:
        circuit: Transpiled quantum circuit
        backend: Target backend with error properties
        layout: Circuit layout information
        
    Returns:
        Sum of (gate_count × error_rate) for all 2-qubit gates
    """
    try:
        # Get backend properties for error rates
        properties = backend.properties()
        if properties is None:
            print("⚠️  Backend properties not available, using uniform error weighting")
            # Fallback: use gate count with uniform error rate
            gate_counts = circuit.count_ops()
            two_qubit_gates = {'cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 'iswap', 'swap'}
            cx_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
            return cx_count * 0.01  # Assume 1% error rate
        
        error_weighted_sum = 0.0
        
        # Iterate through circuit instructions to get actual qubit pairs
        for instruction, qargs, _ in circuit.data:
            if len(qargs) == 2:  # 2-qubit gate
                # Get physical qubit indices with proper layout handling
                phys_q0 = _get_physical_qubit(qargs[0], circuit, layout)
                phys_q1 = _get_physical_qubit(qargs[1], circuit, layout)
                
                # Get error rate for this qubit pair
                error_rate = _get_gate_error_rate(properties, instruction.name, [phys_q0, phys_q1])
                error_weighted_sum += error_rate
        
        return error_weighted_sum
        
    except Exception as e:
        print(f"⚠️  Error computing error-weighted CX: {e}")
        # Fallback to simple gate count
        gate_counts = circuit.count_ops()
        two_qubit_gates = {'cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 'iswap', 'swap'}
        cx_count = sum(gate_counts.get(gate, 0) for gate in two_qubit_gates)
        return cx_count * 0.01


def _compute_error_weighted_total(circuit: QuantumCircuit, backend: Backend,
                                 layout: Optional[Any] = None) -> float:
    """
    Compute total error-weighted gate count including single-qubit gates.
    
    Args:
        circuit: Transpiled quantum circuit
        backend: Target backend with error properties
        layout: Circuit layout information
        
    Returns:
        Sum of (gate_count × error_rate) for all gates
    """
    try:
        properties = backend.properties()
        if properties is None:
            # Fallback to gate count with uniform error rates
            gate_counts = circuit.count_ops()
            total_gates = sum(count for gate, count in gate_counts.items() 
                            if gate not in ['measure', 'barrier'])
            return total_gates * 0.005  # Assume 0.5% average error rate
        
        error_weighted_sum = 0.0
        
        # Iterate through all circuit instructions
        for instruction, qargs, _ in circuit.data:
            if instruction.name in ['measure', 'barrier']:
                continue
                
            # Get physical qubit indices with proper layout handling
            phys_qubits = []
            for qarg in qargs:
                phys_q = _get_physical_qubit(qarg, circuit, layout)
                phys_qubits.append(phys_q)
            
            # Get error rate for this gate and qubits
            error_rate = _get_gate_error_rate(properties, instruction.name, phys_qubits)
            error_weighted_sum += error_rate
        
        return error_weighted_sum
        
    except Exception as e:
        print(f"⚠️  Error computing total error-weighted gates: {e}")
        # Fallback
        gate_counts = circuit.count_ops()
        total_gates = sum(count for gate, count in gate_counts.items() 
                        if gate not in ['measure', 'barrier'])
        return total_gates * 0.005


def _get_physical_qubit(logical_qubit, circuit: QuantumCircuit, layout: Optional[Any] = None) -> int:
    """
    Get the physical qubit index for a logical qubit, handling different layout types.
    
    Args:
        logical_qubit: Logical qubit (Qubit object or int)
        circuit: Quantum circuit for fallback
        layout: Layout object (TranspileLayout, dict, or None)
        
    Returns:
        Physical qubit index
    """
    if layout is None:
        # No layout, use logical indices
        return circuit.find_bit(logical_qubit).index
    
    try:
        # Handle different layout types
        if hasattr(layout, 'get_physical'):
            # TranspileLayout object
            phys_qubit = layout.get_physical(logical_qubit)
            return phys_qubit if phys_qubit is not None else circuit.find_bit(logical_qubit).index
        
        elif hasattr(layout, '__getitem__'):
            # Dictionary-like layout
            logical_index = circuit.find_bit(logical_qubit).index
            if logical_index in layout:
                return layout[logical_index]
            elif logical_qubit in layout:
                return layout[logical_qubit]
            else:
                return logical_index
        
        else:
            # Unknown layout type, use logical index
            return circuit.find_bit(logical_qubit).index
            
    except (KeyError, AttributeError, TypeError) as e:
        # Fallback to logical index
        return circuit.find_bit(logical_qubit).index


def _get_gate_error_rate(properties, gate_name: str, qubits: List[int]) -> float:
    """
    Get error rate for a specific gate on specific qubits.
    
    Args:
        properties: Backend properties object
        gate_name: Name of the gate
        qubits: List of physical qubit indices
        
    Returns:
        Error rate for this gate
    """
    try:
        if len(qubits) == 1:
            # Single-qubit gate
            qubit = qubits[0]
            
            # Try to find exact gate error rate
            for gate_prop in properties.gates:
                if (gate_prop.gate == gate_name and 
                    len(gate_prop.qubits) == 1 and gate_prop.qubits[0] == qubit):
                    return gate_prop.parameters[0].value  # Usually error rate is first parameter
            
            # Fallback: use readout error or default
            if hasattr(properties, 'readout_error'):
                readout_errors = properties.readout_error(qubit)
                if readout_errors:
                    return readout_errors  # Use readout error as proxy
            
            return 0.001  # Default single-qubit error rate (0.1%)
            
        elif len(qubits) == 2:
            # Two-qubit gate
            q0, q1 = sorted(qubits)  # Ensure consistent ordering
            
            # Try to find exact gate error rate
            for gate_prop in properties.gates:
                if (gate_prop.gate == gate_name and 
                    len(gate_prop.qubits) == 2 and 
                    sorted(gate_prop.qubits) == [q0, q1]):
                    return gate_prop.parameters[0].value
            
            # Fallback: try generic 2-qubit gate errors
            for gate_prop in properties.gates:
                if (gate_prop.gate in ['cx', 'ecr'] and 
                    len(gate_prop.qubits) == 2 and 
                    sorted(gate_prop.qubits) == [q0, q1]):
                    return gate_prop.parameters[0].value
            
            return 0.01  # Default 2-qubit error rate (1%)
        
        else:
            # Multi-qubit gate (rare)
            return 0.05 * len(qubits)  # Scale with number of qubits
            
    except Exception as e:
        # Fallback error rates
        if len(qubits) == 1:
            return 0.001
        elif len(qubits) == 2:
            return 0.01
        else:
            return 0.05 * len(qubits)


def _compute_connectivity_score(circuit: QuantumCircuit, backend: Backend) -> float:
    """
    Compute a connectivity score measuring how well the circuit uses the backend topology.
    
    Args:
        circuit: Transpiled quantum circuit
        backend: Target backend
        
    Returns:
        Connectivity score (higher is better)
    """
    try:
        coupling_map = backend.coupling_map
        if coupling_map is None:
            return 1.0  # All-to-all connectivity
        
        # Count 2-qubit gates that use native connections
        native_gates = 0
        total_2q_gates = 0
        
        for instruction, qargs, _ in circuit.data:
            if len(qargs) == 2:
                total_2q_gates += 1
                q0 = circuit.find_bit(qargs[0]).index
                q1 = circuit.find_bit(qargs[1]).index
                
                # Check if this is a native connection
                if [q0, q1] in coupling_map or [q1, q0] in coupling_map:
                    native_gates += 1
        
        if total_2q_gates == 0:
            return 1.0
        
        return native_gates / total_2q_gates
        
    except Exception as e:
        return 0.5  # Default connectivity score


def plot_metrics(results_df: pd.DataFrame, 
                output_dir: str = "results/",
                save_plots: bool = True) -> None:
    """
    Create visualization plots for optimization results.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save plots
        save_plots: Whether to save plots to disk
    """
    # TODO: Create comparison plots (custom vs stock)
    # TODO: Plot CX reduction vs circuit size
    # TODO: Plot compilation time scaling
    # TODO: Create error-weighted metrics plots
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Placeholder plots
    axes[0, 0].set_title("CX Count Comparison")
    axes[0, 1].set_title("Compilation Time")
    axes[1, 0].set_title("Circuit Depth")
    axes[1, 1].set_title("Error-Weighted CX")
    
    plt.tight_layout()
    
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_cx_reduction(custom_metrics: Dict[str, float], 
                        baseline_metrics: Dict[str, float]) -> float:
    """
    Compute CX count reduction percentage.
    
    Args:
        custom_metrics: Metrics from custom layout pass
        baseline_metrics: Metrics from stock transpilation
        
    Returns:
        Percentage reduction in CX count (positive = improvement)
    """
    if not (custom_metrics.get('success', False) and baseline_metrics.get('success', False)):
        return 0.0
    
    custom_cx = custom_metrics.get('cx_count', 0)
    baseline_cx = baseline_metrics.get('cx_count', 0)
    
    if baseline_cx == 0:
        return 0.0
    
    return (1.0 - custom_cx / baseline_cx) * 100.0


def compute_comprehensive_comparison(custom_metrics: Dict[str, float], 
                                   baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute comprehensive comparison metrics between custom and baseline transpilation.
    
    Args:
        custom_metrics: Metrics from custom layout transpilation
        baseline_metrics: Metrics from baseline transpilation
        
    Returns:
        Dictionary with reduction percentages and absolute improvements
    """
    if not (custom_metrics.get('success', False) and baseline_metrics.get('success', False)):
        return {
            'comparison_valid': False,
            'cx_reduction_percent': 0.0,
            'depth_reduction_percent': 0.0,
            'error_weighted_cx_reduction_percent': 0.0,
            'total_gate_reduction_percent': 0.0
        }
    
    def safe_reduction(custom_val, baseline_val):
        """Safely compute reduction percentage."""
        if baseline_val == 0:
            return 0.0 if custom_val == 0 else -100.0  # Negative means increase
        return (1.0 - custom_val / baseline_val) * 100.0
    
    def safe_ratio(custom_val, baseline_val):
        """Safely compute ratio."""
        if baseline_val == 0:
            return float('inf') if custom_val > 0 else 1.0
        return custom_val / baseline_val
    
    return {
        'comparison_valid': True,
        
        # Primary metrics (reduction percentages)
        'cx_reduction_percent': safe_reduction(
            custom_metrics.get('cx_count', 0), 
            baseline_metrics.get('cx_count', 0)
        ),
        'depth_reduction_percent': safe_reduction(
            custom_metrics.get('depth', 0), 
            baseline_metrics.get('depth', 0)
        ),
        'error_weighted_cx_reduction_percent': safe_reduction(
            custom_metrics.get('error_weighted_cx', 0.0), 
            baseline_metrics.get('error_weighted_cx', 0.0)
        ),
        'total_gate_reduction_percent': safe_reduction(
            custom_metrics.get('total_gates', 0), 
            baseline_metrics.get('total_gates', 0)
        ),
        'single_qubit_reduction_percent': safe_reduction(
            custom_metrics.get('single_qubit_count', 0), 
            baseline_metrics.get('single_qubit_count', 0)
        ),
        
        # Absolute improvements
        'absolute_cx_reduction': (
            baseline_metrics.get('cx_count', 0) - custom_metrics.get('cx_count', 0)
        ),
        'absolute_depth_reduction': (
            baseline_metrics.get('depth', 0) - custom_metrics.get('depth', 0)
        ),
        'absolute_gates_reduction': (
            baseline_metrics.get('total_gates', 0) - custom_metrics.get('total_gates', 0)
        ),
        
        # Performance ratios
        'compile_time_ratio': safe_ratio(
            custom_metrics.get('compile_time', 0.0), 
            baseline_metrics.get('compile_time', 0.001)  # Avoid division by zero
        ),
        'connectivity_improvement': (
            custom_metrics.get('connectivity_score', 0.0) - 
            baseline_metrics.get('connectivity_score', 0.0)
        ),
        
        # IBM-specific metrics
        'ecr_reduction_percent': safe_reduction(
            custom_metrics.get('ecr_count', 0), 
            baseline_metrics.get('ecr_count', 0)
        ),
        
        # Quality metrics
        'gate_density_ratio': safe_ratio(
            custom_metrics.get('gate_density', 0.0), 
            baseline_metrics.get('gate_density', 0.001)
        ),
        'depth_efficiency_ratio': safe_ratio(
            custom_metrics.get('depth_efficiency', 0.0), 
            baseline_metrics.get('depth_efficiency', 0.001)
        )
    }


def format_comparison_results(comparison: Dict[str, float], 
                            circuit_name: str = "Circuit") -> str:
    """
    Format comparison results for display.
    
    Args:
        comparison: Comparison metrics from compute_comprehensive_comparison
        circuit_name: Name of the circuit for display
        
    Returns:
        Formatted string with comparison results
    """
    if not comparison.get('comparison_valid', False):
        return f"❌ {circuit_name}: Comparison not valid (transpilation failed)"
    
    # Primary metrics
    cx_reduction = comparison['cx_reduction_percent']
    depth_reduction = comparison['depth_reduction_percent']
    error_reduction = comparison['error_weighted_cx_reduction_percent']
    
    # Format with appropriate emoji indicators
    def format_metric(value, unit='%', threshold=0):
        if value > threshold:
            return f"✅ {value:+.1f}{unit}"
        elif value < -threshold:
            return f"❌ {value:+.1f}{unit}"
        else:
            return f"➖ {value:+.1f}{unit}"
    
    result = f"📊 {circuit_name} Results:\n"
    result += f"   CX gates: {format_metric(cx_reduction)} "
    result += f"({comparison['absolute_cx_reduction']:+d} gates)\n"
    result += f"   Depth: {format_metric(depth_reduction)} "
    result += f"({comparison['absolute_depth_reduction']:+d})\n"
    result += f"   Error-weighted CX: {format_metric(error_reduction)}\n"
    result += f"   Total gates: {format_metric(comparison['total_gate_reduction_percent'])}\n"
    result += f"   Compile time: {comparison['compile_time_ratio']:.2f}x\n"
    
    # Add connectivity improvement if significant
    if abs(comparison['connectivity_improvement']) > 0.01:
        result += f"   Connectivity: {comparison['connectivity_improvement']:+.3f}\n"
    
    return result


def benchmark_summary_stats(results_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute summary statistics across multiple benchmark results.
    
    Args:
        results_list: List of comparison results from batch benchmarking
        
    Returns:
        Dictionary with summary statistics
    """
    if not results_list:
        return {'valid_results': 0}
    
    # Extract valid comparisons
    valid_comparisons = [
        r['comparison'] for r in results_list 
        if r.get('comparison') and r['comparison'].get('comparison_valid', False)
    ]
    
    if not valid_comparisons:
        return {'valid_results': 0}
    
    # Compute statistics for key metrics
    def compute_stats(metric_key):
        values = [c[metric_key] for c in valid_comparisons if metric_key in c]
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        import numpy as np
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return {
        'valid_results': len(valid_comparisons),
        'total_results': len(results_list),
        'cx_reduction': compute_stats('cx_reduction_percent'),
        'depth_reduction': compute_stats('depth_reduction_percent'),
        'error_weighted_cx_reduction': compute_stats('error_weighted_cx_reduction_percent'),
        'compile_time_ratio': compute_stats('compile_time_ratio'),
        
        # Success rate
        'success_rate': len(valid_comparisons) / len(results_list) * 100.0,
        
        # Target achievement (≥25% CX reduction)
        'target_achievement_rate': sum(
            1 for c in valid_comparisons 
            if c.get('cx_reduction_percent', 0) >= 25.0
        ) / len(valid_comparisons) * 100.0 if valid_comparisons else 0.0
    }


def export_results(results: List[Dict[str, Any]], 
                  output_file: str = "results/benchmark_results.csv") -> None:
    """
    Export benchmark results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    df = pd.DataFrame(results)
    Path(output_file).parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")


if __name__ == "__main__":
    # Test basic utility functions
    print("🧪 Testing utility functions...")
    print("=" * 60)
    
    # Test convenience functions
    from qiskit.circuit.library import QuantumVolume
    test_circuit = QuantumVolume(5, depth=3, seed=42)
    
    print(f"📊 Testing convenience functions with {test_circuit.name}...")
    
    # Test interaction_graph convenience function
    graph = interaction_graph(test_circuit)
    print(f"   Interaction graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test load_benchmarks
    circuits = load_benchmarks()
    print(f"   Loaded {len(circuits)} benchmark circuits")
    
    print(f"\n✅ Utility functions working correctly!")
    print(f"🚀 CircuitAnalyzer is now in its own module: circuit_analyzer.py")
    print(f"🚀 BenchmarkRunner is now in its own module: benchmark_runner.py")