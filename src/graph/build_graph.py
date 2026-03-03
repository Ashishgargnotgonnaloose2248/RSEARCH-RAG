import psycopg2
import networkx as nx

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname="RagDb",       # change if needed
    user="postgres",
    password="Ashishgarg22#",   # change if needed
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# -----------------------------
# BUILD DIRECTED CITATION GRAPH
# -----------------------------
G = nx.DiGraph()

# Add paper nodes
cur.execute("SELECT paper_id FROM papers;")
papers = cur.fetchall()

for (paper_id,) in papers:
    G.add_node(paper_id)

# Add citation edges (citing -> cited)
cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
citations = cur.fetchall()

for source, target in citations:
    G.add_edge(source, target)

# -----------------------------
# GRAPH STATISTICS
# -----------------------------
print("\n==============================")
print("CITATION GRAPH ANALYSIS")
print("==============================")

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

print("Graph density:", round(nx.density(G), 6))
print("Weakly connected components:", nx.number_weakly_connected_components(G))

# -----------------------------
# DEGREE ANALYSIS
# -----------------------------
avg_in_degree = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
avg_out_degree = sum(dict(G.out_degree()).values()) / G.number_of_nodes()

print("\nAverage In-Degree:", round(avg_in_degree, 2))
print("Average Out-Degree:", round(avg_out_degree, 2))

# Most cited paper (highest in-degree)
in_degrees = G.in_degree()
most_cited = max(in_degrees, key=lambda x: x[1])

print("\nMost Cited Paper ID:", most_cited[0])
print("Citation Count:", most_cited[1])

# Top 5 by in-degree
top_5_in = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 Most Cited Papers (In-Degree):")
for paper, count in top_5_in:
    print(f"{paper} -> {count}")

# -----------------------------
# PAGERANK (Influence Score)
# -----------------------------
print("\nComputing PageRank...")

pagerank = nx.pagerank(G)

top_5_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 Papers by PageRank:")
for paper, score in top_5_pr:
    print(f"{paper} -> {round(score, 6)}")

# -----------------------------
# CLEANUP
# -----------------------------
cur.close()
conn.close()

print("\nAnalysis Complete.")