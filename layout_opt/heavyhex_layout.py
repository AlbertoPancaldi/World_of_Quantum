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
        layout_dict_qubits = {}
        for logical_idx, physical_idx in layout_dict.items():
            if logical_idx < len(dag.qubits):
                qubit_obj = dag.qubits[logical_idx]
                layout_dict_qubits[qubit_obj] = physical_idx
            else:
                print(f"⚠️  Warning: logical qubit {logical_idx} not found in DAG qubits")
        
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
        
        # Get Heavy-Hex topology analysis results
        hex_clusters = self.analysis_results['hex_clusters']
        distance_matrix = self.analysis_results['distance_matrix']
        
        print(f"📐 Found {len(hex_clusters)} Heavy-Hex clusters for assignment")
        
        layout_dict = {}
        used_physical = set()
        used_clusters = set()
        
        # Sort communities by size (largest first) to assign big communities to hex clusters first
        communities = sorted(communities, key=len, reverse=True)
        
        # Try to assign each community to the best available Heavy-Hex cluster
        for community in communities:
            best_cost = np.inf
            best_cluster = None
            best_assignment = None
            
            # Try each available hex cluster
            for cluster_idx, hex_cluster in enumerate(hex_clusters):
                if cluster_idx in used_clusters:
                    continue
                    
                # Skip if cluster is too small for community
                if len(hex_cluster) < len(community):
                    continue
                
                # Skip if any qubits in cluster are already used
                if any(q in used_physical for q in hex_cluster):
                    continue
                
                # Calculate cost of assigning this community to this cluster
                cost, assignment = self._calculate_community_to_cluster_cost(
                    community, hex_cluster, interaction_graph, distance_matrix
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster_idx
                    best_assignment = assignment
            
            # Assign community to best cluster if found
            if best_cluster is not None:
                hex_cluster = hex_clusters[best_cluster]
                print(f"  📍 Assigned community {community} to hex cluster {hex_cluster[:len(community)]}")
                
                for logical, physical in best_assignment.items():
                    layout_dict[logical] = physical
                    used_physical.add(physical)
                
                used_clusters.add(best_cluster)
            else:
                # No suitable hex cluster found, use greedy individual placement
                print(f"  ⚠️  No hex cluster available for community {community}, using individual placement")
                
                for logical in community:
                    best_physical = self._find_best_individual_placement(
                        logical, interaction_graph, distance_matrix, used_physical, layout_dict
                    )
                    layout_dict[logical] = best_physical
                    used_physical.add(best_physical)
        
        # Assign any remaining logical qubits
        all_available = list(range(self.backend.configuration().n_qubits))
        for logical in range(num_qubits):
            if logical not in layout_dict:
                for physical in all_available:
                    if physical not in used_physical:
                        layout_dict[logical] = physical
                        used_physical.add(physical)
                        break
        
        return layout_dict
    
    def _calculate_community_to_cluster_cost(self,
                                           community: List[int],
                                           hex_cluster: List[int],
                                           interaction_graph: nx.Graph,
                                           distance_matrix: np.ndarray) -> Tuple[float, Dict[int, int]]:
        """Calculate cost of assigning a community to a specific hex cluster."""
        
        # Create assignment (map logical qubits to first N physical qubits in cluster)
        assignment = {}
        for i, logical in enumerate(community):
            if i < len(hex_cluster):
                assignment[logical] = hex_cluster[i]
            else:
                # Community larger than cluster - this shouldn't happen due to pre-filtering
                return np.inf, {}
        
        # Calculate cost using proper border penalty + error penalty
        total_cost = 0.0
        lambda_error = 0.1  # Error penalty weight
        
        # Only penalize interactions that cross cluster boundaries or have significant distance
        for i, q1 in enumerate(community):
            for j, q2 in enumerate(community[i+1:], i+1):
                if interaction_graph.has_edge(q1, q2):
                    weight = interaction_graph[q1][q2]['weight']
                    p1, p2 = assignment[q1], assignment[q2]
                    
                    # Distance penalty (for non-adjacent qubits)
                    distance = distance_matrix[p1, p2]
                    if distance > 1:  # Only penalize non-adjacent placements
                        distance_cost = weight * distance
                    else:
                        distance_cost = 0  # Adjacent qubits have no penalty
                    
                    # CX error penalty
                    cx_error = self._get_cx_error_rate(p1, p2)
                    error_cost = lambda_error * weight * cx_error
                    
                    total_cost += distance_cost + error_cost
        
        return total_cost, assignment
    
    def _get_cx_error_rate(self, physical_qubit1: int, physical_qubit2: int) -> float:
        """Get CX gate error rate between two physical qubits."""
        try:
            if hasattr(self.backend, 'properties') and self.backend.properties():
                properties = self.backend.properties()
                
                # Try to get ECR gate error (IBM's native 2Q gate)
                for gate in properties.gates:
                    if (gate.gate == 'ecr' and 
                        set(gate.qubits) == {physical_qubit1, physical_qubit2}):
                        return gate.parameters[0].value  # gate error
                        
                # Fallback: try CX gate error
                for gate in properties.gates:
                    if (gate.gate == 'cx' and 
                        set(gate.qubits) == {physical_qubit1, physical_qubit2}):
                        return gate.parameters[0].value
                        
                # If no specific gate found, use average single-qubit error as approximation
                qubit_errors = []
                for qubit in [physical_qubit1, physical_qubit2]:
                    if qubit < len(properties.qubits):
                        qubit_props = properties.qubits[qubit]
                        for param in qubit_props:
                            if param.name == 'readout_error':
                                qubit_errors.append(param.value)
                
                if qubit_errors:
                    return sum(qubit_errors) / len(qubit_errors)
                    
        except Exception as e:
            print(f"⚠️  Could not get error rate for qubits {physical_qubit1}-{physical_qubit2}: {e}")
        
        # Default error rate if properties not available
        return 0.01  # 1% default error rate
    
    def _find_best_individual_placement(self,
                                      logical_qubit: int,
                                      interaction_graph: nx.Graph,
                                      distance_matrix: np.ndarray,
                                      used_physical: set,
                                      current_layout: Dict[int, int] = None) -> int:
        """Find best individual placement for a logical qubit."""
        
        if current_layout is None:
            current_layout = {}
        
        best_cost = np.inf
        best_physical = 0
        all_available = list(range(self.backend.configuration().n_qubits))
        
        for physical in all_available:
            if physical in used_physical:
                continue
                
            # Calculate cost of placing this logical qubit at this physical location
            cost = 0.0
            lambda_error = 0.1
            
            # Check interactions with already-placed qubits
            for neighbor in interaction_graph.neighbors(logical_qubit):
                if neighbor in current_layout:
                    neighbor_physical = current_layout[neighbor]
                    weight = interaction_graph[logical_qubit][neighbor]['weight']
                    distance = distance_matrix[physical, neighbor_physical]
                    cx_error = self._get_cx_error_rate(physical, neighbor_physical)
                    
                    cost += weight * distance + lambda_error * weight * cx_error
            
            if cost < best_cost:
                best_cost = cost
                best_physical = physical
        
        return best_physical
    
    def _find_best_community_placement(self,
                                     community: List[int],
                                     interaction_graph: nx.Graph,
                                     distance_matrix: np.ndarray,
                                     available_physical: List[int],
                                     used_physical: set) -> int:
        """Find the best starting physical qubit for a community (legacy method - now unused)."""
        # This method is now replaced by _calculate_community_to_cluster_cost
        # but keeping for compatibility
        
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
        """Calculate the cost of placing a community at a given position (legacy method)."""
        # This method is now replaced by _calculate_community_to_cluster_cost
        # but keeping for compatibility
        
        total_cost = 0.0
        lambda_error = 0.1
        
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
                        # Distance penalty (only for non-adjacent)
                        distance = distance_matrix[p1][p2]
                        distance_cost = weight * distance if distance > 1 else 0
                        
                        # CX error penalty
                        cx_error = self._get_cx_error_rate(p1, p2)
                        error_cost = lambda_error * weight * cx_error
                        
                        total_cost += distance_cost + error_cost
        
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