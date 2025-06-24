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
from qiskit.transpiler import TransformationPass, Layout, PassManager
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
        
        # DEBUG: Show interaction graph details
        print(f"📊 Interaction graph: {interaction_graph.number_of_nodes()} nodes, {interaction_graph.number_of_edges()} edges")
        if interaction_graph.number_of_edges() > 0:
            edge_weights = [data['weight'] for _, _, data in interaction_graph.edges(data=True)]
            print(f"    Total interactions: {sum(edge_weights)}")
            print(f"    Max edge weight: {max(edge_weights)}")
            print(f"    First 5 edges: {list(interaction_graph.edges(data=True))[:5]}")
        
        if interaction_graph.number_of_edges() == 0:
            # No interactions, use trivial layout
            print("⚠️  No interactions found - using trivial layout (this bypasses all Heavy-Hex logic!)")
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
        
        print(f"✅ Layout optimization complete: {len(layout_dict)} qubits assigned")
        return dag
    
    # REMOVED: Broken create_complete_pass_manager method
    # This method was fundamentally broken because it called transpile() inside a transpilation pass,
    # creating recursion and defeating the purpose of custom layout optimization.
    # The correct approach is to use proper PassManager construction externally.
    
    def _build_interaction_graph(self, dag: DAGCircuit) -> nx.Graph:
        """Build weighted interaction graph from circuit."""
        # USE PROPER QISKIT DAG→CIRCUIT CONVERSION
        from qiskit.converters import dag_to_circuit
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
                
            # GET ORIGINAL SCORE
            score = result.get('quantum_score', 0)
            
            # ADD: HEAVY PENALTY FOR OVERSIZED COMMUNITIES  
            communities = result['communities']
            oversized_penalty = 0
            for community in communities:
                if len(community) > 7:  # Max hex cluster size
                    # Massive penalty for communities that can't fit in hex clusters
                    oversized_penalty += (len(community) - 7) * 10.0  # 10x penalty per extra qubit
            
            adjusted_score = score - oversized_penalty
            
            print(f"    🔍 {algo}: original_score={score:.3f}, oversized_penalty={oversized_penalty:.3f}, adjusted_score={adjusted_score:.3f}")
            
            if adjusted_score > best_score:
                best_score = adjusted_score
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
                    community, hex_cluster, interaction_graph, distance_matrix, layout_dict
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
                                           distance_matrix: np.ndarray,
                                           current_global_layout: Dict[int, int]) -> Tuple[float, Dict[int, int]]:
        """
        Calculate cost of assigning a community to a specific hex cluster.
        
        COMPLETE COST FORMULA:
        Cost(C,S;π) = Σ(u<v∈C) w_uv * [max(0, d_π(u)π(v) - 1)]  [internal distance excess]
                    + Σ(u∈C, x∈V_fixed) w_ux * d_π(u)π(x)     [edges to already-placed]  
                    + λ * (internal_error + external_error)
        """
        
        # Create geometric assignment within hex cluster
        assignment = self._create_geometric_assignment(community, hex_cluster, interaction_graph)
        
        if len(assignment) != len(community):
            # Community larger than cluster - this shouldn't happen due to pre-filtering
            return np.inf, {}
        
        lambda_error = 0.1  # Error penalty weight
        
        print(f"    🔍 Evaluating community {community} → hex cluster {hex_cluster[:len(community)]}")
        
        # TERM 1: Internal distance excess penalty - Σ(u<v∈C) w_uv * [max(0, d_π(u)π(v) - 1)]
        internal_distance_cost = 0.0
        internal_error_cost = 0.0
        total_internal_interactions = 0
        
        for i, q1 in enumerate(community):
            for j, q2 in enumerate(community[i+1:], i+1):
                if interaction_graph.has_edge(q1, q2):
                    total_internal_interactions += 1
                    weight = interaction_graph[q1][q2]['weight']
                    p1, p2 = assignment[q1], assignment[q2]
                    
                    # FIXED: Only penalize distance > 1 (excess distance within hex)
                    distance = distance_matrix[p1, p2]
                    distance_excess = max(0, distance - 1)  # ← KEY FIX
                    distance_penalty = weight * distance_excess
                    internal_distance_cost += distance_penalty
                    
                    # Internal error penalty
                    cx_error = self._get_cx_error_rate(p1, p2)
                    internal_error_cost += weight * cx_error
                    
                    print(f"      🔗 Internal: {q1}→{q2} (phys {p1}→{p2}), "
                          f"weight={weight}, distance={distance:.1f}, excess={distance_excess:.1f}, "
                          f"dist_cost={distance_penalty:.2f}, error_cost={weight * cx_error:.3f}")
        
        # TERM 2: External edges cost - Σ(u∈C, x∈V_fixed) w_ux * d_π(u)π(x)
        external_distance_cost = 0.0
        external_error_cost = 0.0
        total_external_interactions = 0
        
        # FIXED: Now actually use current_global_layout
        V_fixed = set(current_global_layout.keys())  # Already-mapped logical qubits
        
        for u in community:
            p_u = assignment[u]  # Physical location of community qubit u
            
            # Check interactions with all already-placed qubits
            for x in V_fixed:
                if interaction_graph.has_edge(u, x):
                    total_external_interactions += 1
                    weight = interaction_graph[u][x]['weight']
                    p_x = current_global_layout[x]  # Physical location of already-placed qubit
                    
                    # External distance cost (no max(0, d-1) here - full distance matters)
                    distance = distance_matrix[p_u, p_x]
                    distance_penalty = weight * distance
                    external_distance_cost += distance_penalty
                    
                    # External error penalty
                    cx_error = self._get_cx_error_rate(p_u, p_x)
                    external_error_cost += weight * cx_error
                    
                    print(f"      🌐 External: {u}→{x} (phys {p_u}→{p_x}), "
                          f"weight={weight}, distance={distance:.1f}, "
                          f"dist_cost={distance_penalty:.2f}, error_cost={weight * cx_error:.3f}")
        
        # TOTAL COST: All three terms
        total_cost = (internal_distance_cost + 
                     external_distance_cost + 
                     lambda_error * (internal_error_cost + external_error_cost))
        
        print(f"    💰 Cost breakdown:")
        print(f"       Internal distance (excess): {internal_distance_cost:.2f}")
        print(f"       External distance: {external_distance_cost:.2f}")
        print(f"       Internal error (λ={lambda_error}): {lambda_error * internal_error_cost:.3f}")
        print(f"       External error (λ={lambda_error}): {lambda_error * external_error_cost:.3f}")
        print(f"       TOTAL: {total_cost:.2f}")
        print(f"    📊 Interactions: {total_internal_interactions} internal, {total_external_interactions} external")
        
        return total_cost, assignment
    
    def _create_geometric_assignment(self, 
                                   community: List[int], 
                                   hex_cluster: List[int],
                                   interaction_graph: nx.Graph) -> Dict[int, int]:
        """
        Create geometry-aware assignment within hex cluster.
        Uses heavy-hex topology structure to place highly connected qubits optimally.
        """
        if len(community) > len(hex_cluster):
            return {}
        
        # STEP 1: Analyze connectivity within community
        connectivity = {}
        edge_weights = {}
        for logical in community:
            degree = 0
            for other in community:
                if other != logical and interaction_graph.has_edge(logical, other):
                    weight = interaction_graph[logical][other]['weight']
                    degree += weight
                    edge_weights[(min(logical, other), max(logical, other))] = weight
            connectivity[logical] = degree
        
        # STEP 2: Analyze geometry within hex cluster using distance matrix
        cluster_distances = {}
        cluster_centrality = {}
        
        # Get distance matrix from topology analyzer
        distance_matrix = self.analysis_results['distance_matrix']
        
        # Calculate centrality of each position in hex cluster
        for i, phys_i in enumerate(hex_cluster):
            centrality = 0.0
            for j, phys_j in enumerate(hex_cluster):
                if i != j and phys_i < len(distance_matrix) and phys_j < len(distance_matrix):
                    dist = distance_matrix[phys_i, phys_j]
                    cluster_distances[(phys_i, phys_j)] = dist
                    # Central positions have lower average distance to others
                    centrality += 1.0 / (1.0 + dist)  # Inverse distance centrality
            cluster_centrality[phys_i] = centrality
        
        # STEP 3: Greedy optimization assignment
        # Most connected logical qubits → most central physical positions
        assignment = {}
        used_physical = set()
        
        # Sort logical qubits by connectivity (highest first)
        sorted_logical = sorted(community, key=lambda q: connectivity[q], reverse=True)
        
        # Sort physical positions by centrality (most central first)  
        sorted_physical = sorted(hex_cluster, key=lambda p: cluster_centrality.get(p, 0), reverse=True)
        
        # For small communities, use simple greedy matching
        if len(community) <= 3:
            for i, logical in enumerate(sorted_logical):
                if i < len(sorted_physical):
                    assignment[logical] = sorted_physical[i]
                    used_physical.add(sorted_physical[i])
        else:
            # For larger communities, use optimization-based assignment
            assignment = self._optimize_hex_assignment(
                community, hex_cluster, interaction_graph, 
                edge_weights, cluster_distances, connectivity, cluster_centrality
            )
        
        return assignment
    
    def _optimize_hex_assignment(self, 
                               community: List[int],
                               hex_cluster: List[int], 
                               interaction_graph: nx.Graph,
                               edge_weights: Dict[Tuple[int, int], float],
                               cluster_distances: Dict[Tuple[int, int], float],
                               connectivity: Dict[int, float],
                               cluster_centrality: Dict[int, float]) -> Dict[int, int]:
        """
        Optimize assignment using local search within hex cluster.
        Minimizes weighted distance cost for high-interaction edges.
        """
        import random
        
        # Initialize with greedy assignment
        assignment = {}
        used_physical = set()
        
        sorted_logical = sorted(community, key=lambda q: connectivity[q], reverse=True)
        sorted_physical = sorted(hex_cluster, key=lambda p: cluster_centrality.get(p, 0), reverse=True)
        
        for i, logical in enumerate(sorted_logical):
            if i < len(sorted_physical):
                assignment[logical] = sorted_physical[i]
                used_physical.add(sorted_physical[i])
        
        # Calculate initial cost
        def calculate_cost(assign):
            cost = 0.0
            for (l1, l2), weight in edge_weights.items():
                if l1 in assign and l2 in assign:
                    p1, p2 = assign[l1], assign[l2]
                    dist = cluster_distances.get((min(p1, p2), max(p1, p2)), 1.0)
                    cost += weight * dist
            return cost
        
        current_cost = calculate_cost(assignment)
        best_assignment = assignment.copy()
        best_cost = current_cost
        
        # Simple local search: try swapping pairs
        random.seed(self.seed)
        for _ in range(20 * len(community)):  # Limited iterations for speed
            if len(community) < 2:
                break
                
            # Pick two random logical qubits to swap
            l1, l2 = random.sample(community, 2)
            if l1 not in assignment or l2 not in assignment:
                continue
                
            # Swap their physical assignments
            p1, p2 = assignment[l1], assignment[l2]
            assignment[l1], assignment[l2] = p2, p1
            
            # Check if this improves cost
            new_cost = calculate_cost(assignment)
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignment = assignment.copy()
            
            # Revert swap
            assignment[l1], assignment[l2] = p1, p2
        
        print(f"      🎯 Hex assignment optimization: {current_cost:.2f} → {best_cost:.2f}")
        return best_assignment
    
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

if __name__ == "__main__":
    print("GreedyCommunityLayout implementation complete!") 