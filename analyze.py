import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data

def load_data(path="trained_data.pt"):
    data = torch.load(path)
    print("Loaded data with", data.num_nodes, "nodes and", data.edge_index.shape[1], "edges.")
    return data

def get_attention_matrix(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weights = data.edge_attr

    attention = torch.zeros((num_nodes, num_nodes))

    for idx in range(edge_index.shape[1]):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        attention[i, j] = edge_weights[idx].item()

    # Apply ReLU to ignore negative influences ?
    attention = torch.nn.functional.relu(attention)

    # Row-normalize to get attention-like scores
    row_sums = attention.sum(dim=1, keepdim=True) + 1e-6
    attention = attention / row_sums

    return attention.numpy()

def plot_attention_heatmap(attention_matrix, title="Attention-Like Edge Weights"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, cmap='viridis')
    plt.title(title)
    plt.xlabel("To node (j)")
    plt.ylabel("From node (i)")
    plt.tight_layout()
    plt.show()

def top_k_influencers(att_matrix, node_idx, k=5):
    scores = att_matrix[node_idx]
    top_k = np.argsort(scores)[-k:][::-1]
    print(f"Top {k} nodes influencing node {node_idx}:")
    for i, node in enumerate(top_k):
        print(f"{i+1}. Node {node} with score {scores[node]:.4f}")
    return top_k

def main():
    data = load_data()
    attention_matrix = get_attention_matrix(data)
    plot_attention_heatmap(attention_matrix)

    # print top influencers for node 42
    top_k_influencers(attention_matrix, node_idx=42, k=5)

if __name__ == "__main__":
    main()
