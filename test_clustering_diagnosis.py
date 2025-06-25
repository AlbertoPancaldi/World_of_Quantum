#!/usr/bin/env python3
"""
Test script to demonstrate the new clustering diagnostic metrics.

This script runs the comprehensive clustering diagnosis on a QuantumVolume
circuit to show exactly what's wrong with the clustering.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qiskit.circuit.library import QuantumVolume
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from layout_opt.utils import build_interaction_graph
from layout_opt.clustering import comprehensive_clustering_diagnosis
from config_loader import load_config

def main():
    print("🧪 Testing New Clustering Diagnostic Metrics")
    print("="*60)
    
    # Load configuration
    load_config()
    
    # Create test circuit
    circuit = QuantumVolume(21, 21, seed=42)
    backend = FakeBrisbane()
    
    print(f"Circuit: QuantumVolume(21, 21, seed=42)")
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    print()
    
    # Build interaction graph
    interaction_graph = build_interaction_graph(circuit)
    
    print(f"Interaction graph: {interaction_graph.number_of_nodes()} nodes, {interaction_graph.number_of_edges()} edges")
    
    # Calculate density
    n_nodes = interaction_graph.number_of_nodes()
    n_edges = interaction_graph.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0
    print(f"Graph density: {density:.1%}")
    print()
    
    # Run comprehensive diagnosis
    diagnosis = comprehensive_clustering_diagnosis(
        interaction_graph, 
        algorithm="greedy_modularity",
        target_cluster_size=7,
        resolution=1.0
    )
    
    print("\n" + "="*60)
    print("🎯 FINAL ASSESSMENT")
    print("="*60)
    print(f"Overall quality: {diagnosis['overall_quality']}")
    
    if diagnosis['issues']:
        print(f"\nKey problems identified:")
        for issue in diagnosis['issues']:
            print(f"  • {issue}")
    
    if diagnosis['recommendations']:
        print(f"\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  • {rec}")
    
    # Test different algorithms
    print("\n" + "="*60)
    print("🔬 ALGORITHM COMPARISON")
    print("="*60)
    
    algorithms = ['greedy_modularity', 'louvain', 'spectral']
    
    for algo in algorithms:
        print(f"\n--- {algo.upper()} ---")
        try:
            diag = comprehensive_clustering_diagnosis(
                interaction_graph,
                algorithm=algo,
                target_cluster_size=7,
                resolution=1.0
            )
            
            metrics = diag['standard_metrics']
            print(f"Summary: {metrics['num_clusters']} clusters, "
                  f"Internal ratio: {metrics['internal_weight_ratio']:.3f}, "
                  f"Max conductance: {metrics['max_conductance']:.3f}")
            
        except Exception as e:
            print(f"❌ {algo} failed: {e}")
    
    print("\n🎉 Diagnostic complete!")
    

if __name__ == "__main__":
    main() 