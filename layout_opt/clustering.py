"""
Graph Clustering Algorithms for Quantum Circuit Layout

Provides multiple clustering approaches for grouping logical qubits based on
their interaction patterns. Designed to work with Heavy-Hex topology optimization.

Available algorithms:
- Greedy modularity optimization (fast, good quality)
- Louvain community detection (higher quality, slower)
- Spectral clustering (geometric insights)
- K-means on adjacency embedding (controllable cluster sizes)
"""

from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
import networkx.algorithms.community as nx_community

from config_loader import get_config


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, seed: Optional[int] = None):
        try:
            config = get_config()
            self.seed = seed or config.get_seed()
        except RuntimeError:
            # Config not loaded, use default seed
            self.seed = seed or 42
        np.random.seed(self.seed)
    
    @abstractmethod
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """
        Detect communities in the interaction graph.
        
        Args:
            graph: Weighted interaction graph (nodes=logical qubits)
            target_cluster_size: Preferred cluster size (None for automatic)
            
        Returns:
            List of communities, each community is a list of node IDs
        """
        pass
    
    def get_algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
        return self.__class__.__name__


class GreedyModularityClustering(ClusteringAlgorithm):
    """
    Fast greedy modularity optimization.
    
    Best for: ≤50ms runtime requirement, good quality clusters
    Trade-offs: May not find global optimum, cluster sizes vary
    """
    
    def __init__(self, seed: Optional[int] = None, resolution: float = 1.0):
        super().__init__(seed)
        self.resolution = resolution
    
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """Use NetworkX greedy modularity optimization."""
        if graph.number_of_nodes() == 0:
            return []
        
        # Use resolution parameter to control cluster sizes
        communities = nx_community.greedy_modularity_communities(
            graph, 
            resolution=self.resolution
        )
        
        # Convert frozensets to lists
        result = [list(community) for community in communities]
        
        # Post-process to respect target cluster size if specified
        if target_cluster_size is not None:
            result = self._adjust_cluster_sizes(result, target_cluster_size)
        
        return result
    
    def _adjust_cluster_sizes(self, communities: List[List[int]], 
                             target_size: int) -> List[List[int]]:
        """Split large clusters and merge small ones."""
        adjusted = []
        
        for community in communities:
            if len(community) <= target_size:
                adjusted.append(community)
            else:
                # Split large cluster
                for i in range(0, len(community), target_size):
                    adjusted.append(community[i:i + target_size])
        
        return adjusted


class LouvainClustering(ClusteringAlgorithm):
    """
    Louvain algorithm for high-quality community detection.
    
    Best for: Higher quality clusters, can handle resolution parameter
    Trade-offs: Slower than greedy modularity, requires python-louvain
    """
    
    def __init__(self, seed: Optional[int] = None, resolution: float = 1.0):
        super().__init__(seed)
        self.resolution = resolution
    
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """Use Louvain algorithm if available, fallback to greedy modularity."""
        try:
            import community as community_louvain
            
            # Convert to format expected by python-louvain
            partition = community_louvain.best_partition(
                graph, 
                resolution=self.resolution, 
                random_state=self.seed
            )
            
            # Group nodes by community ID
            communities_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = []
                communities_dict[comm_id].append(node)
            
            result = list(communities_dict.values())
            
        except ImportError:
            # Fallback to greedy modularity
            print("Warning: python-louvain not available, using greedy modularity")
            fallback = GreedyModularityClustering(self.seed, self.resolution)
            result = fallback.detect_communities(graph, target_cluster_size)
        
        return result


class SpectralGraphClustering(ClusteringAlgorithm):
    """
    Spectral clustering on graph Laplacian.
    
    Best for: Geometric insights, controllable number of clusters
    Trade-offs: Requires predefined number of clusters, O(n³) complexity
    """
    
    def __init__(self, seed: Optional[int] = None, n_clusters: int = 8):
        super().__init__(seed)
        self.n_clusters = n_clusters
    
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """Use spectral clustering on adjacency matrix."""
        if graph.number_of_nodes() == 0:
            return []
        
        # Determine number of clusters
        n_nodes = graph.number_of_nodes()
        if target_cluster_size is not None:
            n_clusters = max(1, n_nodes // target_cluster_size)
        else:
            n_clusters = min(self.n_clusters, n_nodes)
        
        # Build adjacency matrix
        nodes = list(graph.nodes())
        adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes, weight='weight').toarray()
        
        # For spectral clustering, we need a similarity matrix, not adjacency
        # Convert to similarity matrix (higher values = more similar)
        max_weight = adj_matrix.max() if adj_matrix.max() > 0 else 1
        similarity_matrix = adj_matrix / max_weight
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=self.seed,
            n_init=10
        )
        
        labels = clustering.fit_predict(similarity_matrix)
        
        # Group nodes by cluster label
        communities = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            communities[label].append(nodes[i])
        
        # Remove empty clusters
        return [comm for comm in communities if len(comm) > 0]


class KMeansEmbeddingClustering(ClusteringAlgorithm):
    """
    K-means clustering on adjacency matrix embedding.
    
    Best for: Controllable cluster sizes, fast execution
    Trade-offs: May not respect graph structure as well as graph methods
    """
    
    def __init__(self, seed: Optional[int] = None, embedding_dim: int = 8):
        super().__init__(seed)
        self.embedding_dim = embedding_dim
    
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """Embed adjacency matrix and apply K-means."""
        if graph.number_of_nodes() == 0:
            return []
        
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Determine number of clusters
        if target_cluster_size is not None:
            n_clusters = max(1, n_nodes // target_cluster_size)
        else:
            # Default: aim for clusters of size ~7 (Heavy-Hex cell size)
            n_clusters = max(1, n_nodes // 7)
        
        # Build adjacency matrix with weights
        adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes, weight='weight').toarray()
        
        # Dimensionality reduction if needed
        if n_nodes > self.embedding_dim:
            pca = PCA(n_components=self.embedding_dim, random_state=self.seed)
            features = pca.fit_transform(adj_matrix)
        else:
            features = adj_matrix
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.seed,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(features)
        
        # Group nodes by cluster label
        communities = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            communities[label].append(nodes[i])
        
        # Remove empty clusters
        return [comm for comm in communities if len(comm) > 0]


class AdaptiveHeavyHexClustering(ClusteringAlgorithm):
    """
    Heavy-Hex aware clustering that tries to create clusters matching
    the 7-qubit hex and 4-qubit kite structure.
    
    Best for: Heavy-Hex topology optimization, respects hardware constraints
    Trade-offs: More complex, may not work well for other topologies
    """
    
    def __init__(self, seed: Optional[int] = None, prefer_hex: bool = True):
        super().__init__(seed)
        self.prefer_hex = prefer_hex  # Prefer 7-qubit clusters over 4-qubit
    
    def detect_communities(self, graph: nx.Graph, 
                          target_cluster_size: Optional[int] = None) -> List[List[int]]:
        """Create communities targeting Heavy-Hex cell sizes."""
        if graph.number_of_nodes() == 0:
            return []
        
        # First pass: standard modularity detection
        base_clustering = GreedyModularityClustering(self.seed)
        communities = base_clustering.detect_communities(graph)
        
        # Second pass: reshape to Heavy-Hex friendly sizes
        reshaped = []
        leftover_nodes = []
        
        for community in communities:
            if len(community) <= 4:
                reshaped.append(community)  # Small clusters OK as-is
            elif len(community) <= 7:
                reshaped.append(community)  # Perfect for hex cell
            else:
                # Split large community
                if self.prefer_hex:
                    # Try to create 7-qubit clusters
                    for i in range(0, len(community), 7):
                        chunk = community[i:i + 7]
                        if len(chunk) >= 4:  # Minimum viable cluster
                            reshaped.append(chunk)
                        else:
                            leftover_nodes.extend(chunk)
                else:
                    # Try to create 4-qubit clusters
                    for i in range(0, len(community), 4):
                        chunk = community[i:i + 4]
                        if len(chunk) >= 3:  # Allow slightly smaller
                            reshaped.append(chunk)
                        else:
                            leftover_nodes.extend(chunk)
        
        # Handle leftover nodes by creating small clusters or merging
        if leftover_nodes:
            if len(leftover_nodes) >= 3:
                reshaped.append(leftover_nodes)
            elif reshaped:  # Merge with smallest existing cluster
                smallest_idx = min(range(len(reshaped)), key=lambda i: len(reshaped[i]))
                reshaped[smallest_idx].extend(leftover_nodes)
        
        return reshaped


# Factory function for easy algorithm selection
def get_clustering_algorithm(algorithm: str = "greedy_modularity", 
                           **kwargs) -> ClusteringAlgorithm:
    """
    Factory function to create clustering algorithm instances.
    
    Args:
        algorithm: Algorithm name ("greedy_modularity", "louvain", "spectral", 
                   "kmeans", "heavy_hex")
        **kwargs: Algorithm-specific parameters
        
    Returns:
        ClusteringAlgorithm instance
    """
    algorithms = {
        "greedy_modularity": GreedyModularityClustering,
        "louvain": LouvainClustering,
        "spectral": SpectralGraphClustering,
        "kmeans": KMeansEmbeddingClustering,
        "heavy_hex": AdaptiveHeavyHexClustering,
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")
    
    return algorithms[algorithm](**kwargs)


# Clustering Evaluation Metrics
class ClusteringMetrics:
    """
    Comprehensive clustering evaluation metrics for quantum circuit layout.
    
    Provides standard graph clustering metrics plus quantum-specific evaluations.
    """
    
    @staticmethod
    def modularity(graph: nx.Graph, communities: List[List[int]]) -> float:
        """
        Compute modularity score of the clustering.
        
        Args:
            graph: Original interaction graph
            communities: List of communities (node lists)
            
        Returns:
            Modularity score (-1 to 1, higher is better)
        """
        if not communities or graph.number_of_edges() == 0:
            return 0.0
        
        # Convert communities to partition format for NetworkX
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        
        return nx_community.modularity(graph, communities, weight='weight')
    
    @staticmethod
    def silhouette_score(graph: nx.Graph, communities: List[List[int]]) -> float:
        """
        Compute average silhouette score for clustering.
        
        Args:
            graph: Original interaction graph
            communities: List of communities
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        if len(communities) <= 1 or graph.number_of_nodes() == 0:
            return 0.0
        
        try:
            from sklearn.metrics import silhouette_score
            
            # Build distance matrix from graph
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            # Use shortest path distances
            distances = dict(nx.all_pairs_shortest_path_length(graph))
            distance_matrix = np.zeros((n_nodes, n_nodes))
            
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if node_i in distances and node_j in distances[node_i]:
                        distance_matrix[i, j] = distances[node_i][node_j]
                    else:
                        distance_matrix[i, j] = float('inf')
            
            # Create labels array
            labels = np.zeros(n_nodes, dtype=int)
            for comm_id, community in enumerate(communities):
                for node in community:
                    if node in nodes:
                        labels[nodes.index(node)] = comm_id
            
            # Handle infinite distances
            max_finite = np.max(distance_matrix[np.isfinite(distance_matrix)])
            distance_matrix[np.isinf(distance_matrix)] = max_finite + 1
            
            return silhouette_score(distance_matrix, labels, metric='precomputed')
            
        except ImportError:
            # sklearn not available, return 0 (silhouette score not computed)
            print("Warning: sklearn not available, silhouette score set to 0")
            return 0.0
    
    @staticmethod
    def conductance(graph: nx.Graph, communities: List[List[int]]) -> float:
        """
        Compute average conductance of clusters.
        
        Args:
            graph: Original interaction graph
            communities: List of communities
            
        Returns:
            Average conductance (0 to 1, lower is better)
        """
        if not communities:
            return 1.0
        
        total_conductance = 0.0
        valid_communities = 0
        
        for community in communities:
            if len(community) <= 1:
                continue
                
            # Edges within cluster
            internal_edges = 0
            # Edges leaving cluster  
            external_edges = 0
            
            community_set = set(community)
            
            for node in community:
                for neighbor in graph.neighbors(node):
                    weight = graph[node][neighbor].get('weight', 1)
                    if neighbor in community_set:
                        internal_edges += weight
                    else:
                        external_edges += weight
            
            # Each internal edge counted twice
            internal_edges //= 2
            total_edges = internal_edges + external_edges
            
            if total_edges > 0:
                community_conductance = external_edges / total_edges
                total_conductance += community_conductance
                valid_communities += 1
        
        return total_conductance / valid_communities if valid_communities > 0 else 0.0
    
    @staticmethod
    def cluster_statistics(communities: List[List[int]]) -> Dict[str, Any]:
        """
        Compute cluster size and distribution statistics.
        
        Args:
            communities: List of communities
            
        Returns:
            Dictionary with clustering statistics
        """
        if not communities:
            return {
                'num_clusters': 0,
                'total_nodes': 0,
                'avg_cluster_size': 0.0,
                'cluster_sizes': [],
                'size_std': 0.0,
                'coverage': 0.0
            }
        
        sizes = [len(community) for community in communities]
        total_nodes = sum(sizes)
        
        return {
            'num_clusters': len(communities),
            'total_nodes': total_nodes,
            'avg_cluster_size': np.mean(sizes),
            'cluster_sizes': sorted(sizes),
            'size_std': np.std(sizes),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'coverage': 1.0,  # All nodes should be covered
            'size_7_count': sizes.count(7),  # Heavy-hex relevant
            'size_4_count': sizes.count(4),  # Heavy-hex relevant
            'size_distribution': {size: sizes.count(size) for size in set(sizes)}
        }
    
    @staticmethod
    def calinski_harabasz_score(graph: nx.Graph, communities: List[List[int]]) -> float:
        """
        Compute Calinski-Harabasz index (variance ratio criterion).
        
        Args:
            graph: Original interaction graph
            communities: List of communities
            
        Returns:
            Calinski-Harabasz score (higher is better)
        """
        if len(communities) <= 1 or graph.number_of_nodes() <= 1:
            return 0.0
        
        try:
            from sklearn.metrics import calinski_harabasz_score
            
            # Build distance matrix from graph
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            if n_nodes <= 1:
                return 0.0
            
            # Use shortest path distances as coordinates in distance space
            distances = dict(nx.all_pairs_shortest_path_length(graph))
            coordinates = np.zeros((n_nodes, n_nodes))
            
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if node_i in distances and node_j in distances[node_i]:
                        coordinates[i, j] = distances[node_i][node_j]
                    else:
                        coordinates[i, j] = float('inf')
            
            # Handle infinite distances
            max_finite = np.max(coordinates[np.isfinite(coordinates)])
            coordinates[np.isinf(coordinates)] = max_finite + 1
            
            # Create labels array
            labels = np.zeros(n_nodes, dtype=int)
            for comm_id, community in enumerate(communities):
                for node in community:
                    if node in nodes:
                        labels[nodes.index(node)] = comm_id
            
            return calinski_harabasz_score(coordinates, labels)
            
        except ImportError:
            print("Warning: sklearn not available, Calinski-Harabasz score set to 0")
            return 0.0
    
    @staticmethod
    def davies_bouldin_score(graph: nx.Graph, communities: List[List[int]]) -> float:
        """
        Compute Davies-Bouldin index.
        
        Args:
            graph: Original interaction graph
            communities: List of communities
            
        Returns:
            Davies-Bouldin score (lower is better)
        """
        if len(communities) <= 1 or graph.number_of_nodes() <= 1:
            return float('inf')
        
        try:
            from sklearn.metrics import davies_bouldin_score
            
            # Build distance matrix from graph
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            if n_nodes <= 1:
                return float('inf')
            
            # Use shortest path distances as coordinates
            distances = dict(nx.all_pairs_shortest_path_length(graph))
            coordinates = np.zeros((n_nodes, n_nodes))
            
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if node_i in distances and node_j in distances[node_i]:
                        coordinates[i, j] = distances[node_i][node_j]
                    else:
                        coordinates[i, j] = float('inf')
            
            # Handle infinite distances
            max_finite = np.max(coordinates[np.isfinite(coordinates)])
            coordinates[np.isinf(coordinates)] = max_finite + 1
            
            # Create labels array
            labels = np.zeros(n_nodes, dtype=int)
            for comm_id, community in enumerate(communities):
                for node in community:
                    if node in nodes:
                        labels[nodes.index(node)] = comm_id
            
            return davies_bouldin_score(coordinates, labels)
            
        except ImportError:
            print("Warning: sklearn not available, Davies-Bouldin score set to inf")
            return float('inf')

    @staticmethod
    def quantum_layout_score(graph: nx.Graph, communities: List[List[int]], 
                           heavy_hex_cells: Optional[List[List[int]]] = None) -> float:
        """
        Compute quantum circuit layout quality score.
        
        Args:
            graph: Interaction graph
            communities: Detected communities
            heavy_hex_cells: Available Heavy-Hex cells (if known)
            
        Returns:
            Layout quality score (0 to 1, higher is better)
        """
        if not communities or graph.number_of_edges() == 0:
            return 0.0
        
        # Component scores
        mod_score = max(0, ClusteringMetrics.modularity(graph, communities))
        conduct_score = 1.0 - ClusteringMetrics.conductance(graph, communities)
        
        # Size preference (prefer 7-qubit and 4-qubit clusters for Heavy-Hex)
        stats = ClusteringMetrics.cluster_statistics(communities)
        ideal_sizes = [4, 7]  # Heavy-Hex friendly sizes
        size_score = 0.0
        
        for size in stats['cluster_sizes']:
            if size in ideal_sizes:
                size_score += 1.0
            elif size in [3, 5, 6]:  # Acceptable sizes
                size_score += 0.7
            elif size <= 2:  # Too small
                size_score += 0.3
            else:  # Too large
                size_score += 0.5
        
        size_score /= len(stats['cluster_sizes']) if stats['cluster_sizes'] else 1
        
        # Weighted combination
        total_score = 0.4 * mod_score + 0.3 * conduct_score + 0.3 * size_score
        return max(0.0, min(1.0, total_score))


def evaluate_clustering(graph: nx.Graph, communities: List[List[int]], 
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive clustering evaluation.
    
    Args:
        graph: Original interaction graph
        communities: Clustering result to evaluate
        verbose: Whether to print results
        
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Basic statistics
    stats = ClusteringMetrics.cluster_statistics(communities)
    metrics.update(stats)
    
    # Quality metrics
    metrics['modularity'] = ClusteringMetrics.modularity(graph, communities)
    metrics['silhouette'] = ClusteringMetrics.silhouette_score(graph, communities)
    metrics['conductance'] = ClusteringMetrics.conductance(graph, communities)
    metrics['quantum_score'] = ClusteringMetrics.quantum_layout_score(graph, communities)
    
    # Additional sklearn-based metrics
    calinski_raw = ClusteringMetrics.calinski_harabasz_score(graph, communities)
    davies_raw = ClusteringMetrics.davies_bouldin_score(graph, communities)
    
    # Store raw metrics (no normalization as requested)
    metrics['calinski_harabasz'] = calinski_raw
    metrics['davies_bouldin'] = davies_raw
    
    if verbose:
        print(f"📊 Clustering Evaluation")
        print(f"{'='*40}")
        print(f"Clusters: {metrics['num_clusters']}")
        print(f"Total nodes: {metrics['total_nodes']}")
        print(f"Avg size: {metrics['avg_cluster_size']:.1f} ± {metrics['size_std']:.1f}")
        print(f"Size range: {metrics['min_size']} - {metrics['max_size']}")
        print(f"7-qubit clusters: {metrics['size_7_count']}")
        print(f"4-qubit clusters: {metrics['size_4_count']}")
        print(f"")
        print(f"Quality Metrics:")
        print(f"  Modularity: {metrics['modularity']:.3f}")
        print(f"  Silhouette: {metrics['silhouette']:.3f}")
        print(f"  Conductance: {metrics['conductance']:.3f}")
        print(f"  Quantum Score: {metrics['quantum_score']:.3f}")
    
    return metrics


def compare_clustering_algorithms(graph: nx.Graph, algorithms: List[str] = None, 
                                target_cluster_size: int = 7, resolution: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple clustering algorithms on the same graph.
    
    Args:
        graph: Interaction graph to cluster
        algorithms: List of algorithm names to compare
        target_cluster_size: Target cluster size for all algorithms
        
    Returns:
        Dictionary mapping algorithm names to their evaluation metrics
    """
    if algorithms is None:
        algorithms = ['greedy_modularity', 'louvain', 'spectral', 'kmeans', 'heavy_hex']
    
    results = {}
    
    print(f"🔬 Comparing Clustering Algorithms")
    print(f"{'='*60}")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Target cluster size: {target_cluster_size}")
    print()
    
    for algo_name in algorithms:
        try:
            print(f"Testing {algo_name}...")
            
            # Run clustering with appropriate parameters
            clustering_params = {'seed': 42}
            if algo_name in ['greedy_modularity', 'louvain']:
                clustering_params['resolution'] = resolution
            elif algo_name == 'heavy_hex':
                # heavy_hex doesn't use resolution parameter, only prefer_hex
                try:
                    config = get_config()
                    clustering_params['prefer_hex'] = config.get_clustering_prefer_hex()
                except:
                    clustering_params['prefer_hex'] = True
            
            communities = detect_communities(
                graph, 
                algorithm=algo_name, 
                target_cluster_size=target_cluster_size,
                **clustering_params
            )
            
            # Evaluate
            metrics = evaluate_clustering(graph, communities, verbose=False)
            metrics['algorithm'] = algo_name
            metrics['communities'] = communities  # Store the communities
            results[algo_name] = metrics
            
            print(f"  ✅ {algo_name}: {metrics['num_clusters']} clusters, "
                  f"Q={metrics['quantum_score']:.3f}")
            
        except Exception as e:
            print(f"  ❌ {algo_name} failed: {e}")
            results[algo_name] = {'error': str(e), 'algorithm': algo_name}
    
    # Print comparison summary
    print(f"\n📈 Algorithm Comparison Summary:")
    print(f"{'Algorithm':<18} {'Clusters':<8} {'Mod':<6} {'Sil':<6} {'QScore':<7}")
    print(f"{'-'*50}")
    
    for algo_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{algo_name:<18} {metrics['num_clusters']:<8} "
                  f"{metrics['modularity']:<6.3f} {metrics['silhouette']:<6.3f} "
                  f"{metrics['quantum_score']:<7.3f}")
    
    return results


# Utility function for quick clustering
def detect_communities(graph: nx.Graph, 
                      algorithm: str = "greedy_modularity",
                      target_cluster_size: Optional[int] = None,
                      **kwargs) -> List[List[int]]:
    """
    Convenience function for quick community detection.
    
    Args:
        graph: Interaction graph
        algorithm: Clustering algorithm name
        target_cluster_size: Preferred cluster size
        **kwargs: Algorithm-specific parameters
        
    Returns:
        List of communities
    """
    clustering_algo = get_clustering_algorithm(algorithm, **kwargs)
    return clustering_algo.detect_communities(graph, target_cluster_size) 