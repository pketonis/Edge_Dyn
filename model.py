import torch
import torch.nn as nn

class EdgeDyn(nn.Module):
    def __init__(self, input_dim, hidden_dim, tau=2.0, lambda_w=0.0001, dt=0.1, n_steps=10):
        super().__init__()
        self.f_theta = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.tau = tau
        self.lambda_w = lambda_w
        self.dt = dt
        self.n_steps = n_steps

    def forward(self, x, edge_index, edge_attr, eta_plus, eta_minus):
        e = torch.zeros_like(x)
        src, tgt = edge_index

        for _ in range(self.n_steps):
            x_j = x[src]
            w_ij = edge_attr
            messages = torch.zeros_like(x).index_add(0, tgt, w_ij.unsqueeze(1) * x_j)
            dx = self.f_theta(torch.cat([x, messages], dim=1)) - x
            x = x + self.dt * dx

            de = -e / self.tau + x
            e = e + self.dt * de

            e_src = e[src]
            e_tgt = e[tgt]
            x_src = x[src]
            x_tgt = x[tgt]

            dw = eta_plus * (x_src * e_tgt).sum(dim=1) - eta_minus * (x_tgt * e_src).sum(dim=1) - self.lambda_w * w_ij
            edge_attr = edge_attr + self.dt * dw

            # print(f"dw mean: {dw.mean().item():.6f}, std: {dw.std().item():.6f}")

        return x, edge_attr

class EdgeDynModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gnn = EdgeDyn(input_dim, hidden_dim)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x, edge_index, edge_attr, eta_plus, eta_minus):
        x, edge_attr = self.gnn(x, edge_index, edge_attr, eta_plus, eta_minus)
        return self.output_layer(x).squeeze(), edge_attr
