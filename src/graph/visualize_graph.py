import psycopg2
import networkx as nx
import matplotlib.pyplot as plt

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",  # change this
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Create directed graph
G = nx.DiGraph()

# Fetch papers
cur.execute("SELECT paper_id FROM papers;")
papers = cur.fetchall()

for (paper_id,) in papers:
    G.add_node(paper_id)

# Fetch citations
cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
citations = cur.fetchall()

for source, target in citations:
    G.add_edge(source, target)

cur.close()
conn.close()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Convert to undirected to detect connected components
undirected = G.to_undirected()

# Get the largest connected component
largest_component_nodes = max(nx.connected_components(undirected), key=len)

# Create subgraph of largest component
H = G.subgraph(largest_component_nodes)

print("Largest Component Nodes:", H.number_of_nodes())
print("Largest Component Edges:", H.number_of_edges())

# Draw graph
plt.figure(figsize=(10, 8))

pos = nx.spring_layout(H, k=0.3, seed=42)

nx.draw(
    H,
    pos,
    with_labels=False,
    node_size=120,
    arrows=True
)

plt.title("Largest Connected Component - Citation Graph")
plt.show()