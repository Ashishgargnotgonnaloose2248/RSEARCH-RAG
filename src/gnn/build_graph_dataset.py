import psycopg2
import torch
from torch_geometric.data import Data
import numpy as np

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# -----------------------------
# LOAD PAPERS (NODES)
# -----------------------------
cur.execute("""
SELECT paper_id, embedding
FROM papers
WHERE embedding IS NOT NULL
""")

rows = cur.fetchall()

paper_ids = []
embeddings = []

for paper_id, embedding in rows:
    paper_ids.append(paper_id)
    embeddings.append(embedding)

# Convert to tensor
x = torch.tensor(np.array(embeddings), dtype=torch.float)

print("Node feature matrix shape:", x.shape)

# -----------------------------
# CREATE ID → INDEX MAPPING
# -----------------------------
paper_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}

# -----------------------------
# LOAD CITATIONS (EDGES)
# -----------------------------
cur.execute("""
SELECT citing_paper_id, cited_paper_id
FROM citations
""")

edges = []

for citing, cited in cur.fetchall():

    if citing in paper_to_idx and cited in paper_to_idx:
        source = paper_to_idx[citing]
        target = paper_to_idx[cited]

        edges.append([source, target])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print("Edge index shape:", edge_index.shape)

cur.close()
conn.close()

# -----------------------------
# BUILD GRAPH DATASET
# -----------------------------
data = Data(x=x, edge_index=edge_index)

print("\nGraph Dataset Created")
print(data)

# -----------------------------
# SAVE DATASET
# -----------------------------
torch.save(data, "citation_graph.pt")

print("\nGraph saved as citation_graph.pt")

# -----------------------------
# SAVE NODE MAPPING
# -----------------------------
np.save("paper_ids_gnn.npy", np.array(paper_ids))

print("Saved paper ID mapping as paper_ids_gnn.npy")