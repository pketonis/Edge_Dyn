import torch
import torch.nn as nn
import torch.optim as optim
from model import EdgeDynModel
from dataset import load_metr_la_npz_full
from config import get_config
from utils import plot_edge_weight_hist, plot_edge_weight_heatmap
import os
from torch_geometric.data import Data

def train():
    # os.makedirs("plots", exist_ok=True)
    cfg = get_config()

    x_train, y_train, edge_index, edge_attr, eta_plus, eta_minus = load_metr_la_npz_full()
    x_val, y_val, _, _, _, _ = load_metr_la_npz_full(npz_path="data/METR-LA/val.npz")  # same edge structure

    num_nodes = x_train.size(1)
    model = EdgeDynModel(cfg['input_dim'], cfg['hidden_dim'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.MSELoss()

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        for t in range(5000): # adjust the number of training samples as needed
            x = x_train[t].unsqueeze(1)
            y = y_train[t]

            optimizer.zero_grad()
            pred, updated_edge_attr = model(x, edge_index, edge_attr, eta_plus, eta_minus)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            edge_attr = torch.clamp(updated_edge_attr.detach(), -3.0, 3.0)
            total_loss += loss.item()

        avg_train_loss = total_loss / x_train.size(0)

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for t in range(x_val.size(0)):
                x = x_val[t].unsqueeze(1)
                y = y_val[t]
                pred, _ = model(x, edge_index, edge_attr, eta_plus, eta_minus)
                val_loss += loss_fn(pred, y).item()
            avg_val_loss = val_loss / x_val.size(0)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - "
              f"Edge Weight Mean: {edge_attr.mean():.4f}, Std: {edge_attr.std():.4f}")

        if (epoch + 1) % 5 == 0:
            plot_edge_weight_hist(edge_attr, epoch + 1)
            # plot_edge_weight_heatmap(edge_attr, edge_index, num_nodes, epoch + 1)

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    return data


if __name__ == "__main__":
    trained_data = train()
    torch.save(trained_data, "trained_data.pt")