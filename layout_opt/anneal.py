"""
Simulated Annealing Layout Refinement

1-second simulated annealing pass to refine initial layout assignments.
Uses swap moves and community shifts to minimize weighted distance cost.

Algorithm:
- State: permutation (logical → physical qubit mapping)
- Cost: Σ(weight × distance) + λ·noise_penalty
- Moves: swap two qubits or shift entire community
- Schedule: T0=3.0, alpha=0.95, steps=800
"""

from typing import Dict, List, Optional, Tuple
import random
import time
import numpy as np
from qiskit.providers import Backend
from qiskit.transpiler.coupling import CouplingMap
import networkx as nx
from config_loader import get_config


def refine_layout(backend: Backend, 
                  initial_layout: Dict[int, int],
                  interaction_graph: nx.Graph,
                  distance_matrix: np.ndarray = None,
                  communities: List[List[int]] = None,
                  max_time: float = None,
                  seed: Optional[int] = None) -> Dict[int, int]:
    """
    Refine initial layout using simulated annealing.
    
    Args:
        backend: IBM quantum backend
        initial_layout: Initial qubit mapping (logical -> physical)
        interaction_graph: Weighted interaction graph from circuit
        distance_matrix: Pre-computed distance matrix for backend
        communities: Community assignments for community shift moves
        max_time: Maximum annealing time in seconds (default: 1.0s)
        seed: Random seed for reproducibility
        
    Returns:
        Refined layout mapping
    """
    config = get_config()
    
    # Use config values if not provided
    if seed is None:
        seed = config.get_seed()
    if max_time is None:
        annealing_config = config.config.get('layout_optimization', {}).get('annealing', {})
        max_time = annealing_config.get('max_time', 1.0)
    
    # Build distance matrix if not provided
    if distance_matrix is None:
        distance_matrix = _build_distance_matrix(backend)
    
    # Initialize annealer
    annealer = SimulatedAnnealer(
        backend=backend,
        interaction_graph=interaction_graph,
        distance_matrix=distance_matrix,
        communities=communities,
        seed=seed
    )
    
    # Run annealing
    refined_layout = annealer.anneal(initial_layout, max_time)
    
    return refined_layout


class SimulatedAnnealer:
    """Simulated annealing optimizer for qubit layout."""
    
    def __init__(self, 
                 backend: Backend,
                 interaction_graph: nx.Graph,
                 distance_matrix: np.ndarray,
                 communities: List[List[int]] = None,
                 initial_temp: float = None,
                 cooling_rate: float = None,
                 max_steps: int = None,
                 seed: Optional[int] = None):
        """
        Initialize the annealer.
        
        Args:
            backend: Quantum backend
            interaction_graph: Circuit interaction graph
            distance_matrix: Precomputed distance matrix
            communities: Community assignments for shift moves
            initial_temp: Starting temperature (default: 3.0)
            cooling_rate: Temperature decay factor (default: 0.95)
            max_steps: Maximum annealing steps (default: 800)
            seed: Random seed
        """
        config = get_config()
        annealing_config = config.config.get('layout_optimization', {}).get('annealing', {})
        
        self.backend = backend
        self.coupling_map = backend.coupling_map
        self.interaction_graph = interaction_graph
        self.distance_matrix = distance_matrix
        self.communities = communities or []
        
        # Annealing parameters from config
        self.initial_temp = initial_temp or annealing_config.get('initial_temperature', 3.0)
        self.cooling_rate = cooling_rate or annealing_config.get('cooling_rate', 0.95)
        self.max_steps = max_steps or annealing_config.get('max_steps', 800)
        self.noise_penalty_weight = annealing_config.get('noise_penalty_weight', 0.1)
        self.min_temperature = annealing_config.get('min_temperature', 0.01)
        self.progress_steps = annealing_config.get('progress_steps', 200)
        
        # Move probabilities from config
        move_probs = annealing_config.get('move_probabilities', {})
        self.random_swap_prob = move_probs.get('random_swap', 0.7)
        self.community_shift_prob = move_probs.get('community_shift', 0.3)
        
        # Random seed
        self.seed = seed or config.get_seed()
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Cache error rates for performance
        self._error_cache = {}
        self._precompute_error_rates()
        
    def anneal(self, initial_layout: Dict[int, int], max_time: float) -> Dict[int, int]:
        """
        Run simulated annealing optimization.
        
        Args:
            initial_layout: Starting layout
            max_time: Time limit in seconds
            
        Returns:
            Optimized layout
        """
        start_time = time.time()
        
        # Initialize state
        current_layout = initial_layout.copy()
        best_layout = current_layout.copy()
        
        current_cost = self._compute_cost(current_layout)
        best_cost = current_cost
        initial_cost = current_cost
        
        temperature = self.initial_temp
        step = 0
        accepted_moves = 0
        
        print(f"🔥 Starting annealing: initial_cost={initial_cost:.2f}, T={temperature:.2f}")
        
        # Main annealing loop
        while (time.time() - start_time < max_time and 
               step < self.max_steps and 
               temperature > self.min_temperature):
            
            # Generate move
            new_layout = self._generate_move(current_layout)
            new_cost = self._compute_cost(new_layout)
            
            # Accept/reject decision
            cost_delta = new_cost - current_cost
            if self._accept_move(cost_delta, temperature):
                current_layout = new_layout
                current_cost = new_cost
                accepted_moves += 1
                
                # Track best solution
                if current_cost < best_cost:
                    best_layout = current_layout.copy()
                    best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
            step += 1
            
            # Progress logging (configurable interval)
            if step % self.progress_steps == 0:
                acceptance_rate = accepted_moves / step if step > 0 else 0
                print(f"  Step {step}: T={temperature:.3f}, cost={current_cost:.2f}, "
                      f"best={best_cost:.2f}, accept={acceptance_rate:.1%}")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        improvement = ((initial_cost - best_cost) / initial_cost * 100) if initial_cost > 0 else 0
        acceptance_rate = accepted_moves / step if step > 0 else 0
        
        print(f"✅ Annealing complete: {step} steps in {elapsed_time:.2f}s")
        print(f"   Cost improvement: {initial_cost:.2f} → {best_cost:.2f} ({improvement:+.1f}%)")
        print(f"   Acceptance rate: {acceptance_rate:.1%}")
        print(f"   Steps per second: {step/elapsed_time:.0f}")
        
        # Show detailed breakdown if significant improvement
        if improvement > 1.0:
            print(f"   🎉 Significant improvement achieved!")
        elif improvement > 0:
            print(f"   ✅ Minor improvement achieved")
        else:
            print(f"   ⚠️  No improvement found")
        
        return best_layout
    
    def _compute_cost(self, layout: Dict[int, int]) -> float:
        """
        Simple distance-based cost function.
        Cost = Σ(weight × distance) + λ·Σ(gate_error)
        """
        total_cost = 0.0
        
        for u, v, data in self.interaction_graph.edges(data=True):
            if u in layout and v in layout:
                phys_u = layout[u]
                phys_v = layout[v]
                weight = data['weight']
                
                # Distance cost
                distance = self.distance_matrix[phys_u][phys_v]
                distance_cost = weight * distance
                
                # Noise penalty (gate error)
                error_rate = self._get_error_rate(phys_u, phys_v)
                noise_penalty = weight * self.noise_penalty_weight * error_rate
                
                total_cost += distance_cost + noise_penalty
        
        return total_cost
    
    def _generate_move(self, layout: Dict[int, int]) -> Dict[int, int]:
        """
        Generate a neighboring layout via swap or community shift.
        
        Move types:
        1. Random swap (70% probability)
        2. Community shift (30% probability, if communities available)
        """
        new_layout = layout.copy()
        
        # Choose move type based on config probabilities
        if self.communities and random.random() < self.community_shift_prob:
            # Community shift move
            new_layout = self._community_shift_move(new_layout)
        else:
            # Random swap move
            new_layout = self._random_swap_move(new_layout)
        
        return new_layout
    
    def _random_swap_move(self, layout: Dict[int, int]) -> Dict[int, int]:
        """Generate random swap move between two logical qubits."""
        new_layout = layout.copy()
        logical_qubits = list(layout.keys())
        
        if len(logical_qubits) >= 2:
            # Pick two random logical qubits
            q1, q2 = random.sample(logical_qubits, 2)
            
            # Swap their physical assignments
            new_layout[q1], new_layout[q2] = new_layout[q2], new_layout[q1]
        
        return new_layout
    
    def _community_shift_move(self, layout: Dict[int, int]) -> Dict[int, int]:
        """
        Shift entire community to a different location.
        Find unused physical qubits and move community there.
        """
        if not self.communities:
            return self._random_swap_move(layout)
        
        new_layout = layout.copy()
        
        # Pick random community
        community = random.choice(self.communities)
        community_in_layout = [q for q in community if q in layout]
        
        if len(community_in_layout) < 2:
            return self._random_swap_move(layout)
        
        # Find unused physical qubits
        used_physical = set(layout.values())
        available_physical = [q for q in range(self.backend.num_qubits) if q not in used_physical]
        
        if len(available_physical) < len(community_in_layout):
            return self._random_swap_move(layout)
        
        # Move community to new location
        new_physical = random.sample(available_physical, len(community_in_layout))
        for logical, physical in zip(community_in_layout, new_physical):
            new_layout[logical] = physical
        
        return new_layout
    
    def _accept_move(self, cost_delta: float, temperature: float) -> bool:
        """Metropolis acceptance criterion."""
        if cost_delta <= 0:
            return True  # Always accept improvements
        
        if temperature <= 0:
            return False  # Never accept worse moves at zero temperature
        
        probability = np.exp(-cost_delta / temperature)
        return random.random() < probability
    
    def _precompute_error_rates(self):
        """Precompute error rates for all qubit pairs for performance."""
        if not hasattr(self.backend, 'properties') or self.backend.properties() is None:
            # No error data available, use uniform rates
            self._error_cache = {}
            return
            
        properties = self.backend.properties()
        
        # Cache single-qubit and two-qubit error rates
        for i in range(self.backend.num_qubits):
            for j in range(i + 1, self.backend.num_qubits):
                error_rate = self._compute_gate_error(properties, i, j)
                self._error_cache[(i, j)] = error_rate
                self._error_cache[(j, i)] = error_rate  # Symmetric
    
    def _get_error_rate(self, qubit1: int, qubit2: int) -> float:
        """Get cached error rate for qubit pair."""
        if (qubit1, qubit2) in self._error_cache:
            return self._error_cache[(qubit1, qubit2)]
        
        # Fallback: compute on-demand
        if hasattr(self.backend, 'properties') and self.backend.properties() is not None:
            return self._compute_gate_error(self.backend.properties(), qubit1, qubit2)
        
        return 0.01  # Default 1% error rate
    
    def _compute_gate_error(self, properties, qubit1: int, qubit2: int) -> float:
        """Compute gate error rate for a qubit pair."""
        try:
            # Try to get ECR gate error (common on IBM backends)
            gate_error = properties.gate_error('ecr', [qubit1, qubit2])
            return gate_error
        except:
            try:
                # Fallback to CX gate error
                gate_error = properties.gate_error('cx', [qubit1, qubit2])
                return gate_error
            except:
                # Estimate from readout errors if available
                try:
                    readout1 = properties.readout_error(qubit1)
                    readout2 = properties.readout_error(qubit2)
                    return (readout1 + readout2) * 0.5  # Rough approximation
                except:
                    return 0.01  # Default 1% error rate


def _build_distance_matrix(backend: Backend) -> np.ndarray:
    """Build all-pairs shortest path distance matrix for backend topology."""
    n_qubits = backend.num_qubits
    coupling_map = backend.coupling_map
    
    # Initialize distance matrix with infinity
    distances = np.full((n_qubits, n_qubits), np.inf)
    
    # Set diagonal to zero
    np.fill_diagonal(distances, 0)
    
    # Set directly connected qubits to distance 1
    for edge in coupling_map.get_edges():
        i, j = edge[0], edge[1]
        distances[i][j] = 1
        distances[j][i] = 1  # Symmetric
    
    # Floyd-Warshall algorithm for all-pairs shortest paths
    for k in range(n_qubits):
        for i in range(n_qubits):
            for j in range(n_qubits):
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    return distances


if __name__ == "__main__":
    # Simple test with mock data
    print("🔧 Testing simulated annealing...")
    
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    import networkx as nx
    
    # Create test setup
    backend = FakeBrisbane()
    
    # Create simple interaction graph
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=5)
    graph.add_edge(1, 2, weight=3)
    graph.add_edge(2, 3, weight=4)
    
    # Create initial layout
    initial_layout = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Test annealing
    refined_layout = refine_layout(
        backend=backend,
        initial_layout=initial_layout,
        interaction_graph=graph,
        max_time=0.1  # Short test
    )
    
    print(f"✅ Test complete!")
    print(f"   Initial: {initial_layout}")
    print(f"   Refined: {refined_layout}") 