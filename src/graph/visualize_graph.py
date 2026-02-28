import psycopg2
import networkx as nx
import matplotlib.pyplot as plt

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

G = nx.DiGraph()

# Fetch citations only (no need to fetch all nodes separately)
cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
citations = cur.fetchall()

for source, target in citations:
    G.add_edge(source, target)

cur.close()
conn.close()

print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())

# -------------------------------
# Select Top 800 High-Degree Nodes
# -------------------------------

degree_dict = dict(G.degree())
top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:800]

H = G.subgraph(top_nodes)

print("Subgraph Nodes:", H.number_of_nodes())
print("Subgraph Edges:", H.number_of_edges())

# Draw
plt.figure(figsize=(12, 10))

pos = nx.spring_layout(H, k=0.25, seed=42)

nx.draw(
    H,
    pos,
    node_size=15,
    with_labels=False,
    arrows=False
)

plt.title("Top 800 High-Degree Papers - Citation Network")
plt.show()