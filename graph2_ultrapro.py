#!/usr/bin/env python3
"""
graph20_qubits.py
────────────────────────────────────────────────────────────────────────────
• Build an undirected, weighted *interaction graph* from a Qiskit QuantumCircuit
• Plot the quantum circuit (Matplotlib drawer)
• Plot the interaction graph (nodes, edges, edge-weight labels) WITHOUT networkx

Dependencies:  qiskit  ≥ 1.2,  rustworkx,  matplotlib
"""

import random
import rustworkx as rx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build the interaction graph
# ─────────────────────────────────────────────────────────────────────────────
def interaction_graph(circ: QuantumCircuit) -> rx.PyGraph:
    """Return a PyGraph whose edge weights count two-qubit interactions."""
    g = rx.PyGraph()
    g.add_nodes_from(range(circ.num_qubits))

    q2i = {q: i for i, q in enumerate(circ.qubits)}        # qubit → index
    counts: dict[tuple[int, int], int] = {}

    # Use new-style access to avoid deprecation warning
    for instr in circ.data:
        if len(instr.qubits) == 2:                         # any 2-qubit gate
            u, v = sorted(q2i[q] for q in instr.qubits)    # undirected key
            counts[(u, v)] = counts.get((u, v), 0) + 1

    for (u, v), w in counts.items():
        g.add_edge(u, v, w)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Plot the interaction graph (matplotlib-only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_interaction_graph(rwx_graph: rx.PyGraph, title: str | None = None) -> None:
    """
    Draw nodes, edges, and edge-weight labels with matplotlib only.
    Handles both list and dict output from rustworkx.spring_layout().
    """
    # Coordinates from force-directed layout
    pos_raw = rx.spring_layout(rwx_graph, seed=42)

    # Normalise to list[(x, y)]
    if isinstance(pos_raw, dict):
        pos = [tuple(pos_raw[node]) for node in range(rwx_graph.num_nodes())]
    else:                       # list / numpy array
        pos = [tuple(coord) for coord in pos_raw]

    # Split x, y for scatter
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges + weight labels
    for u, v, w in rwx_graph.weighted_edge_list():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1.4, color="black", zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(w), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    # Draw nodes
    ax.scatter(xs, ys, s=500, color="skyblue", edgecolors="black", zorder=2)

    # Node labels
    for idx, (x, y) in enumerate(pos):
        ax.text(x, y, str(idx), fontsize=10, ha="center", va="center", weight="bold")

    if title:
        ax.set_title(title, pad=15)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build a 20-qubit demo circuit
# ─────────────────────────────────────────────────────────────────────────────
def build_demo_circuit(seed: int = 1234) -> QuantumCircuit:
    """
    Create a 20-qubit circuit with diverse single- and two-qubit gates.
    Repeatable with `seed`.
    """
    random.seed(seed)
    qc = QuantumCircuit(20)

    # Layer 1 – single-qubit mix
    for q in range(20):
        qc.h(q)
        qc.rx(0.3 * (q + 1), q)
        if q % 3 == 0:
            qc.s(q)

    qc.barrier()

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

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Demo when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build sizeable circuit
    circuit = build_demo_circuit()

    # 4-A  Plot the quantum circuit
    #fig_circ = circuit.draw("mpl", scale=0.75)
    #fig_circ.suptitle("20-Qubit Quantum Circuit", y=1.02, fontsize=14)
    

    # 4-B  Build & plot the interaction graph
    graph = interaction_graph(circuit)
    plot_interaction_graph(graph, "Interaction Graph (20 Qubits)")