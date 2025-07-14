import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def load_graph(adj_path='data/adj_mx.pkl'):
    with open(adj_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    return sensor_ids, sensor_id_to_ind, csr_matrix(adj_mx)

def visualize_graph(adj_path='data/adj_mx.pkl', show_weights=False, sample_edges=500):
    sensor_ids, _, adj = load_graph(adj_path)
    G = nx.DiGraph()

    for idx, sensor_id in enumerate(sensor_ids):
        G.add_node(idx, label=str(sensor_id))

    # Add edges (optionally sample for clarity)
    row, col = adj.nonzero()
    for i, j in zip(row, col):
        if G.number_of_edges() >= sample_edges:
            break
        weight = adj[i, j]
        G.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(G, seed=42, k=0.3)  # layout with force-directed positioning

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=6)

    plt.title("METR-LA Traffic Graph (Sampled)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("metr_la_graph.png")
    plt.show()

if __name__ == "__main__":
    visualize_graph(show_weights=False)
