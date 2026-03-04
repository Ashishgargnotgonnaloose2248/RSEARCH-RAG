import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_hat):
        # A_hat = normalized adjacency matrix
        return self.linear(torch.matmul(A_hat, X))


class GCN(nn.Module):
    def __init__(self, input_dim=768):
        super(GCN, self).__init__()

        self.gcn1 = GCNLayer(input_dim, 128)
        self.gcn2 = GCNLayer(128, 64)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X, A_hat):
        X = F.relu(self.gcn1(X, A_hat))
        X = F.relu(self.gcn2(X, A_hat))
        out = self.ffn(X)
        return out