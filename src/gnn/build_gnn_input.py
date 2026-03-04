import torch
import networkx as nx
import numpy as np


def normalize_adjacency(G):
    A = nx.to_numpy_array(G)
    I = np.eye(A.shape[0])
    A = A + I  # add self-loops

    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_hat, dtype=torch.float32)