#!/usr/bin/env python3
"""
interaction_graph_plot.py
-------------------------

* Build an undirected, weighted “interaction graph” from a Qiskit QuantumCircuit:
    – Nodes  : logical-qubit indices (0 … n-1)
    – Edges  : any pair of qubits that appear together in a 2-qubit gate
    – Weight : how many times that pair appears together

* Plot the graph with networkx + matplotlib, showing edge-weight labels.
"""

import rustworkx as rx
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import random


# ---------------------------------------------------------------------------
# 1.  Build the interaction graph
# ---------------------------------------------------------------------------
def interaction_graph(circ: QuantumCircuit) -> rx.PyGraph:
    """
    Parameters
    ----------
    circ : QuantumCircuit
        Logical circuit whose 2-qubit interactions you want to count.

    Returns
    -------
    rustworkx.PyGraph
        Simple undirected graph; edge weights = interaction counts.
    """
    g = rx.PyGraph()
    g.add_nodes_from(range(circ.num_qubits))          # one node per logical qubit

    # robust mapping: qubit object -> integer index
    q2i = {q: i for i, q in enumerate(circ.qubits)}

    counts = {}                                       # (u, v) -> #interactions
    for inst, qargs, _ in circ.data:
        if len(qargs) == 2:                           # any 2-qubit gate
            u, v = sorted(q2i[q] for q in qargs)      # undirected key
            counts[(u, v)] = counts.get((u, v), 0) + 1

    for (u, v), w in counts.items():
        g.add_edge(u, v, w)                           # edge weight = count

    return g


# ---------------------------------------------------------------------------
# 2.  Plot the graph (convert to networkx for convenience)
# ---------------------------------------------------------------------------
def plot_interaction_graph(rwx_graph: rx.PyGraph, title: str | None = None) -> None:
    """
    Draw the interaction graph with edge-weight labels.

    Parameters
    ----------
    rwx_graph : rx.PyGraph
        Graph returned by `interaction_graph`.
    title : str | None
        Optional plot title.
    """
    # Copy into networkx (nx handles drawing/labels easily)
    G = nx.Graph()
    G.add_nodes_from(range(rwx_graph.num_nodes()))
    for u, v, w in rwx_graph.weighted_edge_list():
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)      # deterministic layout
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=14)

    # edge labels = weights
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3.  Quick demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example circuit
    qc = QuantumCircuit(20)

    # Layer 1 – single-qubit mix
    for q in range(20):
        qc.h(q)
        qc.rx(0.3 * (q + 1), q)
        if q % 3 == 0:
            qc.s(q)

    qc.barrier()

    for h in range(100):
        qc.cx(1,3)

    # Layer 2 – 90 random entangling gates
    twoq_gates = ("cx", "cz", "swap")
    for _ in range(90):
        a, b = random.sample(range(20), 2)
        gate = random.choice(twoq_gates)
        if gate == "cx":
            qc.cx(a, b)
        elif gate == "cz":
            qc.cz(a, b)
        else:
            qc.swap(a, b)

    qc.barrier()

    # Layer 3 – more single-qubit rotations
    for q in range(20):
        qc.t(q)
        qc.rx(0.15 * q, q)

    print(qc)
    # Build and plot
    graph = interaction_graph(qc)
    plot_interaction_graph(graph, title="Logical-Qubit Interaction Graph")

