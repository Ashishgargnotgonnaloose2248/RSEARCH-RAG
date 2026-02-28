import psycopg2
import networkx as nx

# DB connection
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

print("\nGraph Statistics")
print("----------------------------")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# In-degree analysis (citation importance)
in_degrees = dict(G.in_degree())
max_node = max(in_degrees, key=in_degrees.get)

print("\nMost Cited Paper:")
print("Paper ID:", max_node)
print("Citation Count:", in_degrees[max_node])

# Average degree
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print("\nAverage Degree:", round(avg_degree, 2))

# Largest connected component
largest_component = max(nx.weakly_connected_components(G), key=len)
print("\nLargest Component Size:", len(largest_component))

# Graph density
density = nx.density(G)
print("Graph Density:", density)

# PageRank
pagerank = nx.pagerank(G)
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 Papers by PageRank:")
for paper, score in top_pr:
    print(paper, "->", round(score, 6))