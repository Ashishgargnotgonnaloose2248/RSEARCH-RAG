import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

# -----------------------------
# LOAD GRAPH DATASET
# -----------------------------
data = torch.load("citation_graph.pt")

print("Dataset Loaded:")
print(data)

# -----------------------------
# DEFINE GCN MODEL
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        # Graph Convolution Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Feed Forward Network (FFN)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.ffn(x)

        return x


# -----------------------------
# MODEL INITIALIZATION
# -----------------------------
input_dim = data.x.shape[1]      # 768 (SciBERT embeddings)
hidden_dim = 128
output_dim = 64

model = GCN(input_dim, hidden_dim, output_dim)

print("\nModel Created:")
print(model)

# -----------------------------
# TRAINING SETUP
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

# Dummy training loop (structure only)
for epoch in range(10):

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    # Dummy loss (just to verify pipeline)
    loss = out.mean()

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# -----------------------------
# SAVE TRAINED MODEL
# -----------------------------
torch.save(model.state_dict(), "gnn_model.pt")

print("\nGNN model saved as gnn_model.pt")