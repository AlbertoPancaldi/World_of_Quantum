"""
Utility Functions for Layout Optimization

Benchmarking, metrics computation, and visualization utilities for
evaluating layout optimization performance against stock Qiskit transpilation.

Key functions:
- load_benchmarks(): Load quantum circuits for testing
- transpile_and_score(): Transpile and compute metrics
- plot_metrics(): Visualize optimization results
- CircuitAnalyzer: Build interaction graphs from circuits
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


class CircuitAnalyzer:
    """
    Analyzes quantum circuits to extract interaction patterns for layout optimization.
    
    Converts circuits into weighted interaction graphs where:
    - Nodes = logical qubits  
    - Edges = qubit pairs with 2-qubit gates between them
    - Weights = number of 2-qubit gates between those qubits
    
    Automatically decomposes high-level gates (like QuantumVolume) into elementary gates.
    """
    
    def __init__(self, config=None):
        """Initialize circuit analyzer with configuration."""
        self.config = config or get_config()
        self._two_qubit_gates = ['cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 'iswap', 'swap']
    
    def _decompose_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Decompose circuit into elementary gates for interaction analysis.
        
        Parameters
        ----------
        circuit : QuantumCircuit
            Input circuit that may contain high-level gates
            
        Returns
        -------
        QuantumCircuit
            Decomposed circuit with only elementary gates
        """
        # Check if circuit needs decomposition
        gate_names = [inst.operation.name for inst in circuit.data]
        needs_decomposition = any(name not in ['cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'rzx', 
                                              'iswap', 'swap', 'h', 'x', 'y', 'z', 's', 't', 
                                              'sx', 'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 
                                              'p', 'id', 'barrier', 'measure'] 
                                 for name in gate_names)
        
        if not needs_decomposition:
            return circuit
        
        print(f"🔄 Decomposing {circuit.name or 'circuit'} (contains high-level gates)")
        
        # Create a basic transpilation pass to decompose the circuit
        # We only want decomposition, not optimization or layout
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.providers.fake_provider import GenericBackendV2
        
        # Use a generic backend just for decomposition (we don't care about connectivity here)
        temp_backend = GenericBackendV2(num_qubits=circuit.num_qubits)
        
        # Generate a minimal pass manager that only does decomposition
        pm = generate_preset_pass_manager(
            optimization_level=0,  # No optimization
            backend=temp_backend,
            initial_layout=list(range(circuit.num_qubits))  # Identity layout
        )
        
        # Run decomposition
        decomposed = pm.run(circuit)
        
        print(f"✅ Decomposed: {circuit.depth()} → {decomposed.depth()} depth, "
              f"{len(circuit.data)} → {len(decomposed.data)} gates")
        
        return decomposed
    
    def build_interaction_graph(self, circuit: QuantumCircuit) -> nx.Graph:
        """
        Build weighted interaction graph from quantum circuit.
        
        Automatically decomposes high-level gates before analysis.
        
        Parameters
        ----------
        circuit : QuantumCircuit
            Logical circuit whose 2-qubit interactions you want to count.

        Returns
        -------
        networkx.Graph
            Simple undirected graph; edge weights = interaction counts.
        """
        # Decompose circuit first
        decomposed_circuit = self._decompose_circuit(circuit)
        
        g = nx.Graph()
        g.add_nodes_from(range(decomposed_circuit.num_qubits))  # one node per logical qubit

        # robust mapping: qubit object -> integer index
        q2i = {q: i for i, q in enumerate(decomposed_circuit.qubits)}

        counts = {}  # (u, v) -> #interactions
        for inst, qargs, _ in decomposed_circuit.data:
            if len(qargs) == 2:  # any 2-qubit gate
                u, v = sorted(q2i[q] for q in qargs)  # undirected key
                counts[(u, v)] = counts.get((u, v), 0) + 1

        for (u, v), w in counts.items():
            g.add_edge(u, v, weight=w)  # edge weight = count

        return g
    
    def build_interaction_graph_from_dag(self, dag: DAGCircuit) -> nx.Graph:
        """
        Build weighted interaction graph from DAG circuit.
        
        Args:
            dag: Qiskit DAGCircuit representing the quantum circuit
            
        Returns:
            NetworkX graph with qubits as nodes and gate counts as edge weights
        """
        # Create graph with all logical qubits as nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(dag.num_qubits()))
        
        # Count 2-qubit gates between each pair of qubits
        gate_counts = {}
        
        for node in dag.op_nodes():
            op = node.op
            
            # Only process 2-qubit gates
            if (hasattr(op, 'num_qubits') and op.num_qubits == 2 and 
                op.name.lower() in self._two_qubit_gates):
                
                # Get the logical qubit indices
                qubits = [dag.find_bit(qubit).index for qubit in node.qargs]
                if len(qubits) == 2:
                    qubit_pair = tuple(sorted(qubits))  # Ensure consistent ordering
                    gate_counts[qubit_pair] = gate_counts.get(qubit_pair, 0) + 1
        
        # Add edges with weights
        for (q1, q2), weight in gate_counts.items():
            graph.add_edge(q1, q2, weight=weight)
        
        return graph
    
    def get_interaction_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Get statistics about the interaction graph."""
        if graph.number_of_edges() == 0:
            return {
                'total_interactions': 0,
                'max_interactions': 0,
                'avg_interactions': 0.0,
                'most_connected_qubits': [],
                'interaction_density': 0.0
            }
        
        # Get edge weights
        weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        
        # Find most connected qubits
        qubit_interactions = {}
        for q1, q2, data in graph.edges(data=True):
            weight = data['weight']
            qubit_interactions[q1] = qubit_interactions.get(q1, 0) + weight
            qubit_interactions[q2] = qubit_interactions.get(q2, 0) + weight
        
        most_connected = sorted(qubit_interactions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_interactions': sum(weights),
            'max_interactions': max(weights),
            'avg_interactions': np.mean(weights),
            'most_connected_qubits': most_connected,
            'interaction_density': graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2) if graph.number_of_nodes() > 1 else 0.0
        }
    
    def plot_interaction_graph(self, graph: nx.Graph, title: str = None) -> None:
        """
        Draw the interaction graph with edge-weight labels.

        Parameters
        ----------
        graph : nx.Graph
            Interaction graph from build_interaction_graph().
        title : str | None
            Optional plot title.
        """
        pos = nx.spring_layout(graph, seed=42)  # deterministic layout
        nx.draw_networkx_nodes(graph, pos, node_size=500)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, font_size=14)

        # edge labels = weights
        edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12)

        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


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


class BenchmarkRunner:
    """Runner for systematic benchmarking of layout optimization."""
    
    def __init__(self, backend: Backend, layout_pass: Optional[Any] = None):
        """
        Initialize benchmark runner.
        
        Args:
            backend: Target quantum backend
            layout_pass: Custom layout pass to test
        """
        self.backend = backend
        self.layout_pass = layout_pass
        self.results = []
        
    def run_benchmark_suite(self, circuits: Dict[str, QuantumCircuit]) -> pd.DataFrame:
        """
        Run full benchmark suite comparing custom vs stock transpilation.
        
        Args:
            circuits: Dictionary of test circuits
            
        Returns:
            DataFrame with comparative results
        """
        # TODO: Run both custom and stock transpilation
        # TODO: Compute metrics and reduction percentages
        # TODO: Track timing and performance
        
        results = []
        
        for name, circuit in circuits.items():
            # TODO: Implement actual benchmarking
            result = {
                'circuit_name': name,
                'n_qubits': circuit.num_qubits,
                'custom_cx': 0,
                'baseline_cx': 0,
                'cx_reduction': 0.0,
                'compile_time': 0.0
            }
            results.append(result)
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the CircuitAnalyzer with decomposition
    from qiskit.circuit.library import QuantumVolume
    
    print("🧪 Testing CircuitAnalyzer with Decomposition...")
    print("=" * 60)
    
    # Create a test QuantumVolume circuit (like in your notebook)
    test_circuit = QuantumVolume(5, depth=3, seed=42)
    
    print(f"📊 Original Circuit:")
    print(f"   Name: {test_circuit.name}")
    print(f"   Qubits: {test_circuit.num_qubits}")
    print(f"   Depth: {test_circuit.depth()}")
    print(f"   Gates: {test_circuit.count_ops()}")
    
    # Build interaction graph (with automatic decomposition)
    analyzer = CircuitAnalyzer()
    print(f"\n🔄 Building interaction graph...")
    graph = analyzer.build_interaction_graph(test_circuit)
    
    # Get statistics
    stats = analyzer.get_interaction_statistics(graph)
    
    print(f"\n📈 Interaction Graph Results:")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    print(f"   Total interactions: {stats['total_interactions']}")
    print(f"   Max interactions: {stats['max_interactions']}")
    print(f"   Avg interactions: {stats['avg_interactions']:.2f}")
    print(f"   Most connected qubits: {stats['most_connected_qubits'][:3]}")
    print(f"   Interaction density: {stats['interaction_density']:.3f}")
    
    # Test with a simple circuit that doesn't need decomposition
    print(f"\n🔬 Testing with simple circuit (no decomposition needed)...")
    from qiskit import QuantumCircuit
    simple_circuit = QuantumCircuit(3)
    simple_circuit.cx(0, 1)
    simple_circuit.cx(1, 2)
    simple_circuit.cx(0, 2)
    
    simple_graph = analyzer.build_interaction_graph(simple_circuit)
    simple_stats = analyzer.get_interaction_statistics(simple_graph)
    
    print(f"   Simple circuit edges: {simple_graph.number_of_edges()}")
    print(f"   Simple circuit interactions: {simple_stats['total_interactions']}")
    
    print(f"\n✅ CircuitAnalyzer with decomposition ready for layout optimization!")
    print(f"🚀 Now your QuantumVolume circuits will show proper interaction patterns!") 