"""
Heavy-Hex Topology Analysis and Distance Computation

Comprehensive toolkit for analyzing IBM Heavy-Hex quantum backend topologies.
Provides topology extraction, cluster identification, distance computation,
special qubit detection, and visualization capabilities.

Key Features:
- Backend topology extraction and analysis
- 7-qubit hexagonal cluster identification  
- All-pairs shortest path distance matrices
- Bridge qubit and boundary qubit detection
- Topology visualization helpers
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from qiskit.transpiler.coupling import CouplingMap
from qiskit.providers import Backend
from config_loader import get_config
import logging

# Set up logging
logger = logging.getLogger(__name__)


class TopologyExtractor:
    """Extracts and analyzes basic topology properties from coupling maps."""
    
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        self.graph = self._create_networkx_graph()
        self.n_qubits = coupling_map.size()
        
    def _create_networkx_graph(self) -> nx.Graph:
        """Convert coupling map to NetworkX graph for analysis."""
        G = nx.Graph()
        G.add_nodes_from(range(self.coupling_map.size()))
        G.add_edges_from(self.coupling_map.get_edges())
        return G
    
    def get_degree_distribution(self) -> Dict[int, List[int]]:
        """Get qubits grouped by their degree (connectivity)."""
        degree_groups = {}
        for qubit in range(self.n_qubits):
            degree = self.graph.degree(qubit)
            if degree not in degree_groups:
                degree_groups[degree] = []
            degree_groups[degree].append(qubit)
        return degree_groups
    
    def get_topology_stats(self) -> Dict[str, any]:
        """Get comprehensive topology statistics."""
        degree_dist = self.get_degree_distribution()
        return {
            'n_qubits': self.n_qubits,
            'n_edges': self.graph.number_of_edges(),
            'degree_distribution': {k: len(v) for k, v in degree_dist.items()},
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.n_qubits,
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None,
            'is_connected': nx.is_connected(self.graph)
        }


class HexClusterFinder:
    """
    Identifies 7-qubit hexagonal clusters in Heavy-Hex topology using 
    pattern matching and geometric analysis.
    """
    
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        self.graph = nx.Graph()
        self.graph.add_edges_from(coupling_map.get_edges())
        self._hex_pattern = self._create_ideal_hex_pattern()
        
    def _create_ideal_hex_pattern(self) -> nx.Graph:
        """
        Create the ideal 7‑qubit heavy‑hexagonal pattern.

        In a true heavy‑hex cell the centre qubit (0) has degree 3: it connects
        to every second qubit on the surrounding six‑qubit ring (indices 1–6).
        The ring qubits themselves form a closed hexagonal cycle.

            r1 ─ r2
           /        \
        r0     c     r3
           \        /
            r5 ─ r4
        """
        pattern = nx.Graph()
        centre = 0
        ring = [1, 2, 3, 4, 5, 6]

        # Ring cycle
        for i in range(6):
            pattern.add_edge(ring[i], ring[(i + 1) % 6])

        # Centre connects to alternating ring qubits (degree 3)
        for i in [0, 2, 4]:
            pattern.add_edge(centre, ring[i])

        return pattern
        
    def find_hex_clusters(self) -> List[List[int]]:
        """
        Find 7-qubit hexagonal clusters using hybrid approach.
        
        Strategy:
        1. Try exact pattern matching first (ideal case)
        2. Fall back to degree-based neighborhood clustering
        3. Return practical 7-qubit clusters for layout optimization
        """
        # First try exact pattern matching
        exact_clusters = self._find_exact_hex_patterns()
        
        if exact_clusters:
            logger.info(f"Found {len(exact_clusters)} exact hex patterns")
            return exact_clusters
        
        # Fall back to practical clustering for real Heavy-Hex topologies
        logger.info("No exact hex patterns found, using neighborhood clustering")
        return self._find_hex_neighborhoods()
    
    def _find_exact_hex_patterns(self) -> List[List[int]]:
        """Find exact 7-qubit hex patterns using subgraph isomorphism."""
        hex_clusters = []
        
        try:
            # Find all subgraph isomorphisms of the hex pattern
            matcher = nx.algorithms.isomorphism.GraphMatcher(self.graph, self._hex_pattern)
            
            for mapping in matcher.subgraph_isomorphisms_iter():
                cluster_qubits = sorted(mapping.keys())
                hex_clusters.append(cluster_qubits)
                
            # Remove duplicates
            unique_clusters = []
            seen_clusters = set()
            
            for cluster in hex_clusters:
                cluster_tuple = tuple(cluster)
                if cluster_tuple not in seen_clusters:
                    unique_clusters.append(cluster)
                    seen_clusters.add(cluster_tuple)
                    
            return unique_clusters
            
        except Exception as e:
            logger.warning(f"Exact pattern matching failed: {e}")
            return []
    
    def _find_hex_neighborhoods(self) -> List[List[int]]:
        """
        Find 7-qubit neighborhoods around degree-3 qubits.
        
        This is a practical approach for real Heavy-Hex topologies where
        perfect hex patterns don't exist as discrete subgraphs.
        """
        hex_clusters = []
        degree_3_qubits = [q for q in self.graph.nodes() if self.graph.degree(q) == 3]
        used_qubits = set()
        
        for center in degree_3_qubits:
            if center in used_qubits:
                continue
                
            # Build a 7-qubit neighborhood around this degree-3 qubit
            neighborhood = self._build_neighborhood(center, target_size=7)
            
            if len(neighborhood) == 7 and self._is_valid_cluster(neighborhood):
                hex_clusters.append(sorted(neighborhood))
                # Mark qubits as used (but allow some overlap)
                used_qubits.update(neighborhood[:4])  # Only mark center + 3 others as used
                
        return hex_clusters
    
    def _build_neighborhood(self, center: int, target_size: int) -> List[int]:
        """Build a connected neighborhood of target_size around center qubit."""
        neighborhood = {center}
        queue = [center]
        
        while queue and len(neighborhood) < target_size:
            current = queue.pop(0)
            
            # Add neighbors in order of preference (degree-2 and degree-3 qubits)
            neighbors = list(self.graph.neighbors(current))
            neighbors.sort(key=lambda q: (self.graph.degree(q), q))  # Prefer lower degree
            
            for neighbor in neighbors:
                if neighbor not in neighborhood and len(neighborhood) < target_size:
                    neighborhood.add(neighbor)
                    queue.append(neighbor)
                    
                if len(neighborhood) >= target_size:
                    break
                    
        return list(neighborhood)
    
    def _is_valid_cluster(self, qubits: List[int]) -> bool:
        """Validate that qubits form a reasonable cluster for layout optimization."""
        if len(qubits) != 7:
            return False
            
        subgraph = self.graph.subgraph(qubits)
        
        # Must be connected
        if not nx.is_connected(subgraph):
            return False
            
        # Should have reasonable edge density (not too sparse)
        num_edges = subgraph.number_of_edges()
        if num_edges < 6:  # Minimum for connected 7-node graph
            return False
            
        # Check degree distribution - should be mostly degree 2-3 qubits
        valid_degree_count = sum(1 for q in qubits 
                               if self.graph.degree(q) in [2, 3])
        
        return valid_degree_count >= 5  # At least 5 of 7 should be degree 2-3
    
    def get_cluster_centers(self, hex_clusters: List[List[int]]) -> List[int]:
        """
        Extract the center qubit from each hex cluster.
        
        In the ideal hex pattern, the center is the qubit connected to all 6 others.
        """
        centers = []
        
        for cluster in hex_clusters:
            # Find the qubit with degree 3 within this cluster
            subgraph = self.graph.subgraph(cluster)
            center_candidates = [q for q in cluster if subgraph.degree(q) == 3]
            
            if len(center_candidates) == 1:
                centers.append(center_candidates[0])
            elif len(center_candidates) > 1:
                # Multiple degree-3 qubits - pick first one
                logger.warning(f"Multiple center candidates in cluster {cluster}")
                centers.append(center_candidates[0])
            else:
                # No degree-3 qubit found - use highest degree as fallback
                max_degree_qubit = max(cluster, key=lambda q: subgraph.degree(q))
                centers.append(max_degree_qubit)
                logger.warning(f"No degree-3 center found in cluster {cluster}, using {max_degree_qubit}")
                
        return centers


class DistanceMatrixComputer:
    """Computes all-pairs shortest path distances for topology analysis."""
    
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        # Use an **undirected** graph because geometric distance is direction‑agnostic
        # in Heavy‑Hex devices.  Add *all* qubits first so isolated qubits are not lost.
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(coupling_map.size()))          # NEW
        self.graph.add_edges_from(coupling_map.get_edges())
        self._distance_matrix = None
        
    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute all-pairs shortest path distance matrix.
        
        Returns:
            Distance matrix where entry (i,j) is shortest path from qubit i to j
        """
        if self._distance_matrix is not None:
            return self._distance_matrix
            
        n_qubits = self.coupling_map.size()
        distance_matrix = np.full((n_qubits, n_qubits), np.inf, dtype=float)
        np.fill_diagonal(distance_matrix, 0)  # Distance to self
        
        try:
            # Iterate over the generator to avoid materialising the whole dict
            for src, lengths in nx.all_pairs_shortest_path_length(self.graph):
                for dst, dist in lengths.items():
                    distance_matrix[src, dst] = dist
        except Exception as e:
            logger.warning(f"Error computing distances: {e}")
            # Fallback: set direct neighbors to distance 1
            for edge in self.coupling_map.get_edges():
                i, j = edge
                distance_matrix[i][j] = 1
                distance_matrix[j][i] = 1
            # Run Floyd‑Warshall to complete the matrix
            for k in range(n_qubits):
                for i in range(n_qubits):
                    if distance_matrix[i, k] == np.inf:
                        continue
                    for j in range(n_qubits):
                        if distance_matrix[k, j] == np.inf:
                            continue
                        new_dist = distance_matrix[i, k] + distance_matrix[k, j]
                        if new_dist < distance_matrix[i, j]:
                            distance_matrix[i, j] = new_dist
        
        self._distance_matrix = distance_matrix
        # Return a *copy* so external callers cannot mutate the cached instance.
        return distance_matrix.copy()
    
    def get_distance(self, qubit_i: int, qubit_j: int) -> float:
        """Get distance between two specific qubits."""
        if self._distance_matrix is None:
            self.compute_distance_matrix()
        return self._distance_matrix[qubit_i][qubit_j]


class SpecialQubitDetector:
    """
    Detects special qubits in Heavy-Hex topology:
    - Cluster centers: degree-6 qubits at the center of hexagonal clusters
    - Bridge qubits: degree-2 qubits connecting different clusters  
    - Boundary qubits: degree-1 qubits at the edges of the topology
    """
    
    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map
        self.graph = nx.Graph()
        self.graph.add_edges_from(coupling_map.get_edges())
        self.hex_finder = HexClusterFinder(coupling_map)
        
    def detect_all_special_qubits(self) -> Dict[str, List[int]]:
        """Detect all categories of special qubits."""
        hex_clusters = self.hex_finder.find_hex_clusters()
        
        return {
            'cluster_centers': self._find_cluster_centers(hex_clusters),
            'bridge_qubits': self._find_bridge_qubits(hex_clusters),
            'boundary_qubits': self._find_boundary_qubits(),
            'hex_clusters': hex_clusters
        }
    
    def _find_cluster_centers(self, hex_clusters: List[List[int]]) -> List[int]:
        """
        Find center qubits of hexagonal clusters.
        
        Uses the HexClusterFinder's method to properly identify centers
        from the 7-qubit hex patterns.
        """
        return self.hex_finder.get_cluster_centers(hex_clusters)
    
    def _find_bridge_qubits(self, hex_clusters: List[List[int]]) -> List[int]:
        """
        Find bridge qubits that connect different hex clusters.
        
        Bridge qubits are typically degree-2 qubits that:
        1. Are not part of any hex cluster
        2. Connect qubits from different hex clusters
        3. Act as inter-cluster communication channels
        """
        if not hex_clusters:
            return []
            
        cluster_qubits = set()
        for cluster in hex_clusters:
            cluster_qubits.update(cluster)
            
        bridges = []
        
        # Check all qubits not in clusters
        for qubit in range(self.coupling_map.size()):
            if qubit in cluster_qubits:
                continue
                
            # Bridge qubits are typically degree-2 in Heavy-Hex
            if self.graph.degree(qubit) != 2:
                continue
                
            neighbors = list(self.graph.neighbors(qubit))
            
            # Find which clusters the neighbors belong to
            neighbor_cluster_ids = set()
            for neighbor in neighbors:
                for cluster_id, cluster in enumerate(hex_clusters):
                    if neighbor in cluster:
                        neighbor_cluster_ids.add(cluster_id)
                        
            # Bridge if it connects qubits from different clusters
            if len(neighbor_cluster_ids) >= 2:
                bridges.append(qubit)
                
        return bridges
    
    def _find_boundary_qubits(self) -> List[int]:
        """Find boundary qubits with limited connectivity."""
        boundary = []
        for qubit in range(self.coupling_map.size()):
            degree = self.graph.degree(qubit)
            if degree == 1:  # True boundary qubits
                boundary.append(qubit)
        return boundary


class TopologyVisualizer:
    """Visualization helpers for Heavy-Hex topology analysis."""
    
    def __init__(self, coupling_map: CouplingMap, backend: Backend = None, config=None):
        self.coupling_map = coupling_map
        self.backend = backend
        self.config = config or get_config()
        self.graph = nx.Graph()
        self.graph.add_edges_from(coupling_map.get_edges())
        
    def plot_topology_overview(self, special_qubits: Dict[str, List[int]] = None, 
                              save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive topology overview with special qubits highlighted.
        
        Args:
            special_qubits: Dict from SpecialQubitDetector.detect_all_special_qubits()
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Use real chip coordinates if available, otherwise spring layout
        if self.backend:
            print(f"Backend provided: {self.backend.name}")
            config = self.backend.configuration()
            print(f"Backend config attributes: {dir(config)}")
            if hasattr(config, "qubit_coordinates"):
                print("Backend has qubit_coordinates, using real chip coordinates")
                coords = config.qubit_coordinates
                print(f"Sample coordinates: {list(coords.items())[:5]}")
                # NetworkX expects {node: (x, y)}
                pos = {q: coords[q] for q in self.graph.nodes() if q in coords}
            else:
                print("Backend has no qubit_coordinates attribute, using spring layout")
                pos = nx.spring_layout(self.graph, seed=self.config.get_seed())
        else:
            print("No backend provided, using spring layout")
            pos = nx.spring_layout(self.graph, seed=self.config.get_seed())
        
        # Draw base graph
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, ax=ax)
        
        # Color qubits by type
        if special_qubits:
            # Draw different qubit types with different colors
            centers = special_qubits.get('cluster_centers', [])
            bridges = special_qubits.get('bridge_qubits', [])
            boundaries = special_qubits.get('boundary_qubits', [])
            
            # Regular qubits
            regular = [q for q in self.graph.nodes() 
                      if q not in centers + bridges + boundaries]
            
            if regular:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=regular, 
                                     node_color='lightblue', node_size=100, ax=ax)
            if centers:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=centers,
                                     node_color='red', node_size=200, ax=ax, label='Cluster Centers')
            if bridges:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=bridges,
                                     node_color='orange', node_size=150, ax=ax, label='Bridge Qubits')
            if boundaries:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=boundaries,
                                     node_color='gray', node_size=80, ax=ax, label='Boundary Qubits')
        else:
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                                 node_size=100, ax=ax)
        
        # Add labels for small graphs
        if self.coupling_map.size() <= 30:
            nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)
            
        ax.set_title(f'Heavy-Hex Topology Overview ({self.coupling_map.size()} qubits)')
        if special_qubits:
            ax.legend()
        ax.axis('off')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_hex_clusters(self, hex_clusters: List[List[int]], 
                         save_path: str = None) -> plt.Figure:
        """Plot identified hexagonal clusters with highlighting."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Use real chip coordinates if available, otherwise spring layout
        if self.backend:
            print(f"Backend provided: {self.backend.name}")
            config = self.backend.configuration()
            if hasattr(config, "qubit_coordinates"):
                print("Backend has qubit_coordinates, using real chip coordinates")
                coords = config.qubit_coordinates
                pos = {q: coords[q] for q in self.graph.nodes() if q in coords}
            else:
                print("Backend has no qubit_coordinates attribute, using spring layout")
                pos = nx.spring_layout(self.graph, seed=self.config.get_seed())
        else:
            print("No backend provided, using spring layout")
            pos = nx.spring_layout(self.graph, seed=self.config.get_seed())
        
        # Draw base graph
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, ax=ax)
        
        # Draw regular qubits
        all_cluster_qubits = set()
        for cluster in hex_clusters:
            all_cluster_qubits.update(cluster)
            
        regular_qubits = [q for q in self.graph.nodes() if q not in all_cluster_qubits]
        if regular_qubits:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_qubits,
                                 node_color='lightgray', node_size=50, ax=ax)
        
        # Draw each hex cluster with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(hex_clusters)))
        
        for i, cluster in enumerate(hex_clusters):
            nx.draw_networkx_nodes(self.graph, pos, nodelist=cluster,
                                 node_color=[colors[i]], node_size=150, 
                                 ax=ax, label=f'Hex Cluster {i+1}')
            
            # Highlight cluster edges
            cluster_edges = [(u, v) for u, v in self.graph.edges() 
                           if u in cluster and v in cluster]
            nx.draw_networkx_edges(self.graph, pos, edgelist=cluster_edges,
                                 edge_color=colors[i], width=2, ax=ax)
        
        ax.set_title(f'Heavy-Hex Clusters ({len(hex_clusters)} clusters identified)')
        if len(hex_clusters) <= 10:  # Only show legend for reasonable number of clusters
            ax.legend()
        ax.axis('off')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class HeavyHexTopologyAnalyzer:
    """
    Main orchestrator for Heavy-Hex topology analysis.
    
    Integrates all topology analysis components with config-driven behavior.
    """
    
    def __init__(self, backend: Backend, config=None):
        """
        Initialize comprehensive topology analyzer.
        
        Args:
            backend: IBM quantum backend with Heavy-Hex topology
            config: Configuration object (uses global config if None)
        """
        self.backend = backend
        self.config = config or get_config()
        self.coupling_map = backend.coupling_map
        
        # Initialize analysis components
        self.extractor = TopologyExtractor(self.coupling_map)
        self.cluster_finder = HexClusterFinder(self.coupling_map)
        self.distance_computer = DistanceMatrixComputer(self.coupling_map)
        self.special_detector = SpecialQubitDetector(self.coupling_map)
        self.visualizer = TopologyVisualizer(self.coupling_map, self.backend, self.config)
        
        # Cache for expensive computations
        self._analysis_cache = {}
        
    def analyze_topology(self, cache_results: bool = True) -> Dict[str, any]:
        """
        Run comprehensive topology analysis.
        
        Args:
            cache_results: Whether to cache results for reuse
            
        Returns:
            Complete topology analysis results
        """
        if cache_results and 'full_analysis' in self._analysis_cache:
            return self._analysis_cache['full_analysis']
            
        logger.info("Starting comprehensive Heavy-Hex topology analysis...")
        
        # Basic topology statistics
        topology_stats = self.extractor.get_topology_stats()
        
        # Hexagonal cluster identification
        hex_clusters = self.cluster_finder.find_hex_clusters()
        
        # Distance matrix computation
        distance_matrix = self.distance_computer.compute_distance_matrix()
        
        # Special qubit detection
        special_qubits = self.special_detector.detect_all_special_qubits()
        
        analysis_results = {
            'backend_name': self.backend.name,
            'topology_stats': topology_stats,
            'hex_clusters': hex_clusters,
            'distance_matrix': distance_matrix,
            'special_qubits': special_qubits,
            'analysis_summary': {
                'n_hex_clusters': len(hex_clusters),
                'n_cluster_centers': len(special_qubits['cluster_centers']),
                'n_bridge_qubits': len(special_qubits['bridge_qubits']),
                'n_boundary_qubits': len(special_qubits['boundary_qubits']),
                'avg_cluster_size': np.mean([len(c) for c in hex_clusters]) if hex_clusters else 0,
                'topology_efficiency': self._compute_topology_efficiency(distance_matrix)
            }
        }
        
        if cache_results:
            self._analysis_cache['full_analysis'] = analysis_results
            
        logger.info(f"Topology analysis complete: {analysis_results['analysis_summary']}")
        return analysis_results
    
    def _compute_topology_efficiency(self, distance_matrix: np.ndarray) -> float:
        """Compute topology efficiency metric."""
        finite_distances = distance_matrix[distance_matrix < np.inf]
        if len(finite_distances) == 0:
            return 0.0
        return 1.0 / np.mean(finite_distances) if np.mean(finite_distances) > 0 else 0.0
    
    def get_distance_matrix(self) -> np.ndarray:
        """Get cached distance matrix."""
        return self.distance_computer.compute_distance_matrix()
    
    def get_hex_clusters(self) -> List[List[int]]:
        """Get identified hexagonal clusters."""
        return self.cluster_finder.find_hex_clusters()
    
    def get_special_qubits(self) -> Dict[str, List[int]]:
        """Get all special qubit categories."""
        return self.special_detector.detect_all_special_qubits()
    
    def create_topology_visualizations(self, output_dir: str = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive topology visualizations.
        
        Args:
            output_dir: Directory to save plots (uses config if None)
            
        Returns:
            Dictionary of created figures
        """
        if output_dir is None:
            output_dir = self.config.get_output_directory()
            
        special_qubits = self.get_special_qubits()
        hex_clusters = self.get_hex_clusters()
        
        figures = {}
        print("Creating topology visualizations..")
        # Topology overview
        overview_path = f"{output_dir}/topology_overview.png" if output_dir else None
        figures['overview'] = self.visualizer.plot_topology_overview(
            special_qubits, overview_path)
        
        # Hex clusters
        if hex_clusters:
            clusters_path = f"{output_dir}/hex_clusters.png" if output_dir else None
            figures['clusters'] = self.visualizer.plot_hex_clusters(
                hex_clusters, clusters_path)
        
        return figures


def compute_border_penalty(community: List[int], 
                          cell: List[int], 
                          interaction_graph: nx.Graph,
                          distance_matrix: np.ndarray) -> float:
    """
    Compute penalty for placing a community in a Heavy-Hex cell.
    
    Args:
        community: List of logical qubits in the community
        cell: List of physical qubits in the Heavy-Hex cell
        interaction_graph: Weighted interaction graph
        distance_matrix: Precomputed distance matrix
        
    Returns:
        Border penalty cost
    """
    if len(community) != len(cell):
        raise ValueError("Community and cell must have same size")
        
    total_penalty = 0.0
    
    # Calculate penalty for each logical qubit pair
    for i, logical_i in enumerate(community):
        for j, logical_j in enumerate(community):
            if i < j and interaction_graph.has_edge(logical_i, logical_j):
                # Get interaction strength
                weight = interaction_graph[logical_i][logical_j].get('weight', 1.0)
                
                # Get physical distance
                physical_i = cell[i]
                physical_j = cell[j]
                distance = distance_matrix[physical_i, physical_j]
                
                # Add penalty for non-adjacent qubits
                if distance > 1:
                    total_penalty += weight * distance
                    
    return total_penalty


if __name__ == "__main__":
    print("Heavy-Hex topology analysis module loaded successfully!")
    
    # Example usage
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        backend = FakeBrisbane()
        
        analyzer = HeavyHexTopologyAnalyzer(backend)
        results = analyzer.analyze_topology()
        
        print(f"Analysis complete for {results['backend_name']}:")
        print(f"  - {results['analysis_summary']['n_hex_clusters']} hex clusters found")
        print(f"  - {results['analysis_summary']['n_cluster_centers']} cluster centers")
        print(f"  - {results['analysis_summary']['n_bridge_qubits']} bridge qubits")
        print(f"  - Topology efficiency: {results['analysis_summary']['topology_efficiency']:.3f}")
        
    except ImportError:
        print("Qiskit runtime not available for testing") 