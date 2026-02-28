import psycopg2
import networkx as nx

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",  # change if needed
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Create graph
G = nx.DiGraph()

# Add ALL papers first
cur.execute("SELECT paper_id FROM papers;")
for (paper_id,) in cur.fetchall():
    G.add_node(paper_id)

# Add edges
cur.execute("SELECT citing_paper_id, cited_paper_id FROM citations;")
for source, target in cur.fetchall():
    G.add_edge(source, target)

print("Total Nodes in Graph:", G.number_of_nodes())
print("Total Edges in Graph:", G.number_of_edges())

# Compute PageRank
pagerank_scores = nx.pagerank(G, alpha=0.85)

print("PageRank computed.")

# ---------------------------
# Add column if not exists
# ---------------------------

cur.execute("""
    ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS pagerank_score FLOAT;
""")

# ---------------------------
# Store scores in database
# ---------------------------

for paper_id, score in pagerank_scores.items():
    cur.execute(
        "UPDATE papers SET pagerank_score = %s WHERE paper_id = %s;",
        (score, paper_id)
    )

conn.commit()

conn.commit()

cur.close()
conn.close()

print("PageRank stored successfully.")