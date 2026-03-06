import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ----------------------------
# LOAD GRAPH DATA
# ----------------------------
data = torch.load("citation_graph.pt")

# ----------------------------
# DEFINE SAME MODEL AS TRAINING
# ----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Feed Forward Network
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


# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
input_dim = data.x.shape[1]
hidden_dim = 128
output_dim = 64   # IMPORTANT FIX

model = GCN(input_dim, hidden_dim, output_dim)

model.load_state_dict(torch.load("gnn_model.pt"))
model.eval()

# ----------------------------
# GENERATE NODE EMBEDDINGS
# ----------------------------
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

print("Generated GNN embeddings shape:", embeddings.shape)

# ----------------------------
# SAVE EMBEDDINGS
# ----------------------------
torch.save(embeddings, "gnn_embeddings.pt")

print("Saved as gnn_embeddings.pt")