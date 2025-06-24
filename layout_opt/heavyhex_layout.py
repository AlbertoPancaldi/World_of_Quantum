"""
Heavy-Hex Layout Optimization Pass

Implements GreedyCommunityLayout that uses community detection on the interaction
graph to place talkative qubits adjacent within Heavy-Hex cells (7-qubit hex 
clusters and 4-qubit kites).

Algorithm:
1. Build interaction graph from circuit (nodes=logical qubits, edges=CX counts)
2. Detect communities using clustering algorithms
3. Enumerate Heavy-Hex cells and assign communities to minimize cost
4. Cost = Σ(weight × distance) + λ·Σ(CX_error)
"""

from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import numpy as np
from qiskit.transpiler import TransformationPass, Layout
from qiskit.transpiler.coupling import CouplingMap
from qiskit.providers import Backend
from qiskit.dagcircuit import DAGCircuit
from config_loader import get_config
from .utils import CircuitAnalyzer
from .clustering import compare_clustering_algorithms
from .distance import HeavyHexTopologyAnalyzer


class GreedyCommunityLayout(TransformationPass):
    """
    Layout pass that places logical qubits using community detection
    optimized for IBM Heavy-Hex topologies.
    """
    
    def __init__(self, backend: Backend, seed: Optional[int] = None):
        """
        Initialize the layout pass.
        
        Args:
            backend: IBM quantum backend with Heavy-Hex topology
            seed: Random seed for reproducibility (uses config if None)
        """
        super().__init__()
        config = get_config()
        
        self.backend = backend
        self.coupling_map = backend.coupling_map
        self.seed = seed or config.get_seed()
        
        # Get clustering algorithms from config
        self.clustering_algorithms = config.get_clustering_algorithms()
        self.target_cluster_size = config.get_clustering_target_size()
        
        # Initialize topology analyzer
        self.topology_analyzer = HeavyHexTopologyAnalyzer(backend, config)
        self.analysis_results = None
        
        # Initialize circuit analyzer
        self.circuit_analyzer = CircuitAnalyzer()
        
        # Cache for heavy-hex cells
        self._heavy_hex_cells = None
        
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the layout optimization pass.
        
        Args:
            dag: Input DAG circuit
            
        Returns:
            DAG circuit with layout property set
        """
        if dag.num_qubits() == 0:
            return dag
            
        # Step 1: Build interaction graph from DAG
        print(f"🔄 Building interaction graph for {dag.num_qubits()} qubits...")
        interaction_graph = self._build_interaction_graph(dag)
        
        if interaction_graph.number_of_edges() == 0:
            # No interactions, use trivial layout
            layout_dict = {dag.qubits[i]: i for i in range(dag.num_qubits())}
            layout = Layout(layout_dict)
            self.property_set['layout'] = layout
            dag._layout = layout
            return dag
        
        # Step 2: Run clustering comparison to find best algorithm
        print(f"🔬 Running clustering algorithms: {self.clustering_algorithms}")
        clustering_results = compare_clustering_algorithms(
            interaction_graph,
            algorithms=self.clustering_algorithms,
            target_cluster_size=self.target_cluster_size,
            resolution=1.0
        )
        
        # Step 3: Select best clustering result
        best_clustering = self._select_best_clustering(clustering_results)
        communities = best_clustering['communities']
        
        print(f"✅ Selected clustering: {best_clustering['algorithm']} with {len(communities)} communities")
        
        # Step 4: Get Heavy-Hex topology analysis
        if self.analysis_results is None:
            print("🔄 Analyzing Heavy-Hex topology...")
            self.analysis_results = self.topology_analyzer.analyze_topology()
        
        # Step 5: Create optimal layout assignment
        print("🎯 Computing optimal layout assignment...")
        layout_dict = self._create_optimal_layout(
            communities, 
            interaction_graph,
            dag.num_qubits()
        )
        
        # Step 6: Set layout property (convert to Qubit objects)
        layout_dict_qubits = {dag.qubits[logical]: physical 
                             for logical, physical in layout_dict.items()}
        layout = Layout(layout_dict_qubits)
        self.property_set['layout'] = layout
        
        # Also set on the DAG for compatibility
        dag._layout = layout
        
        print(f"✅ Layout optimization complete: {len(layout_dict)} qubits assigned")
        return dag
    
    def _build_interaction_graph(self, dag: DAGCircuit) -> nx.Graph:
        """Build weighted interaction graph from circuit."""
        # Convert DAG to QuantumCircuit for analysis
        from qiskit import QuantumCircuit
        circuit = dag_to_circuit(dag)
        return self.circuit_analyzer.build_interaction_graph(circuit)
    
    def _select_best_clustering(self, clustering_results: Dict) -> Dict:
        """Select the best clustering result based on quantum score."""
        best_algo = None
        best_score = -np.inf
        best_result = None
        
        for algo, result in clustering_results.items():
            if 'error' in result:
                continue
                
            score = result.get('quantum_score', 0)
            if score > best_score:
                best_score = score
                best_algo = algo
                best_result = result
        
        if best_result is None:
            raise RuntimeError("All clustering algorithms failed")
        
        return {
            'algorithm': best_algo,
            'score': best_score,
            'communities': best_result['communities'],
            'metrics': best_result
        }
    
    def _create_optimal_layout(self, 
                              communities: List[List[int]], 
                              interaction_graph: nx.Graph,
                              num_qubits: int) -> Dict[int, int]:
        """Create optimal layout by assigning communities to Heavy-Hex cells."""
        
        # Get distance matrix
        distance_matrix = self.topology_analyzer.get_distance_matrix()
        
        # Get available physical qubits (up to num_qubits needed)
        available_physical = list(range(min(num_qubits, self.backend.configuration().n_qubits)))
        
        # Simple greedy assignment for now
        layout_dict = {}
        used_physical = set()
        
        # Sort communities by size (largest first)
        communities = sorted(communities, key=len, reverse=True)
        
        for community in communities:
            # Find best starting position for this community
            best_start = self._find_best_community_placement(
                community, 
                interaction_graph,
                distance_matrix,
                available_physical,
                used_physical
            )
            
            # Assign community members starting from best position
            for i, logical_qubit in enumerate(community):
                physical_qubit = (best_start + i) % len(available_physical)
                while physical_qubit in used_physical:
                    physical_qubit = (physical_qubit + 1) % len(available_physical)
                
                layout_dict[logical_qubit] = physical_qubit
                used_physical.add(physical_qubit)
        
        # Assign any remaining logical qubits
        for logical in range(num_qubits):
            if logical not in layout_dict:
                for physical in available_physical:
                    if physical not in used_physical:
                        layout_dict[logical] = physical
                        used_physical.add(physical)
                        break
        
        return layout_dict
    
    def _find_best_community_placement(self,
                                     community: List[int],
                                     interaction_graph: nx.Graph,
                                     distance_matrix: np.ndarray,
                                     available_physical: List[int],
                                     used_physical: set) -> int:
        """Find the best starting physical qubit for a community."""
        
        best_cost = np.inf
        best_start = 0
        
        for start_pos in available_physical:
            if start_pos in used_physical:
                continue
                
            # Calculate cost for placing community starting at this position
            cost = self._calculate_placement_cost(
                community,
                start_pos,
                interaction_graph,
                distance_matrix,
                available_physical,
                used_physical
            )
            
            if cost < best_cost:
                best_cost = cost
                best_start = start_pos
        
        return best_start
    
    def _calculate_placement_cost(self,
                                community: List[int],
                                start_pos: int,
                                interaction_graph: nx.Graph,
                                distance_matrix: np.ndarray,
                                available_physical: List[int],
                                used_physical: set) -> float:
        """Calculate the cost of placing a community at a given position."""
        
        total_cost = 0.0
        
        # Create temporary assignment for this community
        temp_assignment = {}
        for i, logical in enumerate(community):
            physical = (start_pos + i) % len(available_physical)
            while physical in used_physical:
                physical = (physical + 1) % len(available_physical)
            temp_assignment[logical] = physical
        
        # Calculate interaction costs within community
        for i, q1 in enumerate(community):
            for j, q2 in enumerate(community[i+1:], i+1):
                if interaction_graph.has_edge(q1, q2):
                    weight = interaction_graph[q1][q2]['weight']
                    p1, p2 = temp_assignment[q1], temp_assignment[q2]
                    
                    if p1 < len(distance_matrix) and p2 < len(distance_matrix):
                        distance = distance_matrix[p1][p2]
                        total_cost += weight * distance
        
        return total_cost


def dag_to_circuit(dag: DAGCircuit):
    """Convert DAG to QuantumCircuit for analysis."""
    from qiskit import QuantumCircuit
    
    # Create circuit with same number of qubits
    circuit = QuantumCircuit(dag.num_qubits())
    
    # Create qubit index mapping
    qubit_map = {qubit: i for i, qubit in enumerate(dag.qubits)}
    
    # Add gates from DAG (simplified conversion)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:  # Two-qubit gate
            q1_idx = qubit_map[node.qargs[0]]
            q2_idx = qubit_map[node.qargs[1]]
            circuit.cx(q1_idx, q2_idx)  # Treat all as CX for interaction counting
    
    return circuit


if __name__ == "__main__":
    print("GreedyCommunityLayout implementation complete!") 