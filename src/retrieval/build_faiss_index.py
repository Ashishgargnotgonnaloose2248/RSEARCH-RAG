import psycopg2
import numpy as np
import faiss

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
# LOAD EMBEDDINGS
# -----------------------------
cur.execute("SELECT paper_id, embedding FROM papers;")
rows = cur.fetchall()

paper_ids = []
embeddings = []

for paper_id, embedding in rows:
    if embedding is not None:
        paper_ids.append(paper_id)
        embeddings.append(embedding)

cur.close()
conn.close()

embeddings = np.array(embeddings).astype("float32")

print("Loaded embeddings shape:", embeddings.shape)

# -----------------------------
# NORMALIZE (cosine similarity)
# -----------------------------
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# -----------------------------
# BUILD FAISS INDEX
# -----------------------------
index = faiss.IndexFlatIP(dimension)  # inner product

index.add(embeddings)

print("FAISS index built.")
print("Total vectors indexed:", index.ntotal)

# -----------------------------
# SAVE INDEX
# -----------------------------
faiss.write_index(index, "faiss_index.bin")
np.save("paper_ids.npy", np.array(paper_ids))

print("Index saved successfully.")