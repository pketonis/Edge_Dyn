import numpy as np
import torch
from scipy.sparse import csr_matrix
import pickle

def load_metr_la_npz_full(npz_path='data/METR-LA/train.npz', adj_path='data/adj_mx.pkl'):
    data = np.load(npz_path)
    x_seq = data['x'][:, -1, :, 0]  # [23974, 207]
    y_seq = data['y'][:, 0, :, 0]   # [23974, 207]

    x_seq = torch.tensor(x_seq / 100.0, dtype=torch.float32)  # [samples, nodes]
    y_seq = torch.tensor(y_seq / 100.0, dtype=torch.float32)

    num_samples, num_nodes = x_seq.shape

    with open(adj_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    A = csr_matrix(adj_mx)
    edge_index = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)

    edge_attr = torch.randn(edge_index.size(1))
    eta_plus = torch.full((edge_index.size(1),), 0.5)
    eta_minus = torch.full((edge_index.size(1),), 0.5)

    return x_seq, y_seq, edge_index, edge_attr, eta_plus, eta_minus
