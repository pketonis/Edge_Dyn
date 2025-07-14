import matplotlib.pyplot as plt
import torch

def plot_edge_weight_hist(edge_attr, epoch):
    edge_attr = edge_attr.detach().cpu().numpy()
    plt.hist(edge_attr, bins=30, alpha=0.7)
    plt.title(f"Edge Weight Distribution at Epoch {epoch}")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"edge_weights_epoch_{epoch}.png")
    plt.close()

def plot_edge_weight_heatmap(edge_attr, edge_index, num_nodes, epoch):
    edge_matrix = torch.zeros((num_nodes, num_nodes))
    for idx in range(edge_index.size(1)):
        i, j = edge_index[:, idx]
        edge_matrix[i, j] = edge_attr[idx]

    plt.figure(figsize=(8, 6))
    plt.imshow(edge_matrix.cpu(), cmap='bwr', interpolation='nearest')
    plt.colorbar(label='Edge Weight')
    plt.title(f"Edge Weight Heatmap - Epoch {epoch}")
    plt.xlabel("Target Node j")
    plt.ylabel("Source Node i")
    plt.tight_layout()
    plt.savefig(f"plots/heatmap_epoch_{epoch:03}.png")
    plt.close()