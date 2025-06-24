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
    Transpile circuit and compute performance metrics.
    
    Args:
        circuit: Input quantum circuit
        backend: Target quantum backend
        layout_pass: Custom layout pass (None for stock Qiskit)
        optimization_level: Qiskit optimization level
        
    Returns:
        Dictionary of performance metrics
    """
    start_time = time.time()
    config = get_config()
    opt_level = optimization_level or config.get_optimization_level()
    
    if layout_pass is not None:
        # TODO: Use custom layout pass
        # TODO: Combine with stock routing/optimization
        pass_manager = PassManager()
        pass_manager.append(layout_pass)
        # TODO: Add SabreSwap and optimization passes
        transpiled = pass_manager.run(circuit)
    else:
        # Use stock Qiskit transpilation with config parameters
        transpiled = transpile(
            circuit, 
            backend=backend, 
            optimization_level=opt_level,
            seed_transpiler=config.get_seed()
        )
    
    compile_time = time.time() - start_time
    
    # TODO: Compute comprehensive metrics
    metrics = {
        'cx_count': 0,  # TODO: Count CX/ECR gates
        'single_qubit_count': 0,  # TODO: Count single-qubit gates
        'depth': transpiled.depth(),
        'compile_time': compile_time,
        'error_weighted_cx': 0.0,  # TODO: Weight by gate error rates
        'n_qubits': transpiled.num_qubits
    }
    
    return metrics


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
        CX reduction percentage (positive = improvement)
    """
    custom_cx = custom_metrics['cx_count']
    baseline_cx = baseline_metrics['cx_count']
    
    if baseline_cx == 0:
        return 0.0
    
    return (1.0 - custom_cx / baseline_cx) * 100.0


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