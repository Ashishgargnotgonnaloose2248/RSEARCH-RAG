import psycopg2
import networkx as nx

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="RagDb",   # change if needed
    user="postgres",
    password="Ashishgarg22#", # change if needed
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Create directed graph
G = nx.DiGraph()

# Fetch papers (nodes)
cur.execute("SELECT paper_id FROM papers;")
papers = cur.fetchall()

for (paper_id,) in papers:
    G.add_node(paper_id)

# Fetch citations (edges)
cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
citations = cur.fetchall()

for source, target in citations:
    G.add_edge(source, target)

print("Graph Analysis")
print("------------------------")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Most cited paper (highest in-degree)
in_degrees = G.in_degree()
most_cited = max(in_degrees, key=lambda x: x[1])

print("Most cited paper:", most_cited[0])
print("Citation count:", most_cited[1])

# Average degree
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print("Average degree:", round(avg_degree, 2))

# Top 5 cited papers
top_5 = sorted(in_degrees, key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 Most Cited Papers:")
for paper, count in top_5:
    print(paper, "->", count)

cur.close()
conn.close()