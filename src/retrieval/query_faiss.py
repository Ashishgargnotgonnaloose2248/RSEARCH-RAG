import numpy as np
import faiss
import torch
import psycopg2
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# CONFIG
# -----------------------------
ALPHA = 0.6   # semantic weight
BETA = 0.2    # pagerank weight
GAMMA = 0.2   # gnn weight

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
# LOAD GNN EMBEDDINGS
# -----------------------------
gnn_embeddings = torch.load("gnn_embeddings.pt").numpy()
gnn_paper_ids = np.load("paper_ids_gnn.npy")

print("Loaded GNN embeddings:", gnn_embeddings.shape)

# Map paper_id → index
gnn_id_to_index = {pid: i for i, pid in enumerate(gnn_paper_ids)}

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_query(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    embedding = embedding.cpu().numpy().astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def search(query, top_k=5):

    query_vec = embed_query(query)

    # Retrieve semantic candidates
    scores, indices = index.search(query_vec, 50)

    results = []

    for idx, score in zip(indices[0], scores[0]):

        paper_id = paper_ids[idx]
        similarity = float(score)

        results.append((paper_id, similarity))

    # -----------------------------
    # FETCH PAGERANK
    # -----------------------------
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

    # -----------------------------
    # NORMALIZE PAGERANK
    # -----------------------------
    pr_values = list(pagerank_dict.values())

    max_pr = max(pr_values) if pr_values else 1
    min_pr = min(pr_values) if pr_values else 0

    # -----------------------------
    # COMPUTE GNN SCORES
    # -----------------------------
    gnn_scores = []

    for pid in paper_id_list:
        idx = gnn_id_to_index.get(pid)
        if idx is not None:
            gnn_scores.append(np.linalg.norm(gnn_embeddings[idx]))

    max_gnn = max(gnn_scores) if gnn_scores else 1
    min_gnn = min(gnn_scores) if gnn_scores else 0

    # -----------------------------
    # HYBRID SCORING
    # -----------------------------
    hybrid_results = []

    for paper_id, similarity in results:

        # PageRank
        pagerank = pagerank_dict.get(paper_id, 0.0)
        normalized_pr = (pagerank - min_pr) / (max_pr - min_pr + 1e-8)

        # GNN importance
        gnn_idx = gnn_id_to_index.get(paper_id)

        if gnn_idx is not None:
            gnn_vector = gnn_embeddings[gnn_idx]
            gnn_score = np.linalg.norm(gnn_vector)
        else:
            gnn_score = 0.0

        normalized_gnn = (gnn_score - min_gnn) / (max_gnn - min_gnn + 1e-8)

        # Final hybrid score
        final_score = (
            ALPHA * similarity +
            BETA * normalized_pr +
            GAMMA * normalized_gnn
        )

        hybrid_results.append(
            (
                paper_id,
                final_score,
                similarity,
                normalized_pr,
                normalized_gnn
            )
        )

    # -----------------------------
    # SORT RESULTS
    # -----------------------------
    hybrid_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Hybrid Results:\n")

    for i in range(top_k):

        paper_id, final_score, sim, pr, gnn = hybrid_results[i]

        print(
            "Paper ID:", paper_id,
            "| Hybrid:", round(final_score, 4),
            "| Semantic:", round(sim, 4),
            "| PageRank:", round(pr, 4),
            "| GNN:", round(gnn, 4)
        )


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    user_query = input("Enter your query: ")

    search(user_query)

    cur.close()
    conn.close()