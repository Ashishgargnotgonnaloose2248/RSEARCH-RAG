import numpy as np
import faiss
import torch
import psycopg2
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# CONFIG
# -----------------------------
ALPHA = 0.7  # semantic weight
BETA = 0.3   # pagerank weight

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
# LOAD MODEL (SciBERT)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    use_safetensors=True
).to(device)
model.eval()

# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
index = faiss.read_index("faiss_index.bin")
paper_ids = np.load("paper_ids.npy")

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)

    return embedding


# -----------------------------
# SEARCH FUNCTION (HYBRID)
# -----------------------------
def search(query, top_k=5):

    query_vec = embed_query(query)

    # Get more candidates first
    scores, indices = index.search(query_vec, 50)

    results = []

    for idx, score in zip(indices[0], scores[0]):
        paper_id = paper_ids[idx]
        similarity = float(score)
        results.append((paper_id, similarity))

    # Fetch pagerank scores
    paper_id_list = [r[0] for r in results]

    cur.execute(
        """
        SELECT paper_id, pagerank_score
        FROM papers
        WHERE paper_id = ANY(%s)
        """,
        (paper_id_list,)
    )

    pagerank_dict = dict(cur.fetchall())

    # Hybrid scoring
    hybrid_results = []

    for paper_id, similarity in results:
        pagerank = pagerank_dict.get(paper_id, 0.0)

        final_score = ALPHA * similarity + BETA * pagerank

        hybrid_results.append((paper_id, final_score, similarity, pagerank))

    # Re-rank
    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Hybrid Results:\n")

    for i in range(top_k):
        paper_id, final_score, sim, pr = hybrid_results[i]
        print(
            "Paper ID:", paper_id,
            "| Hybrid:", round(final_score, 4),
            "| Semantic:", round(sim, 4),
            "| PageRank:", round(pr, 6)
        )


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search(user_query)

    cur.close()
    conn.close()