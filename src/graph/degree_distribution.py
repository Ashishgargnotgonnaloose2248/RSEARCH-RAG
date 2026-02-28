import psycopg2
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Connect to DB
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

G = nx.DiGraph()

cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
edges = cur.fetchall()

for source, target in edges:
    G.add_edge(source, target)

cur.close()
conn.close()

print("Graph Loaded")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -------- IN-DEGREE (Citation Count) --------

in_degrees = [deg for node, deg in G.in_degree()]

# Count frequency of each degree
degree_counts = np.bincount(in_degrees)

degrees = np.arange(len(degree_counts))

# Remove zero entries
nonzero = degree_counts > 0
degrees = degrees[nonzero]
degree_counts = degree_counts[nonzero]

# -------- PLOT --------

plt.figure(figsize=(8, 6))
plt.loglog(degrees, degree_counts)

plt.xlabel("Citation Count (In-Degree)")
plt.ylabel("Number of Papers")
plt.title("Degree Distribution (Log-Log Scale)")
plt.show()