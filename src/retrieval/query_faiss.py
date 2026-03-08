import numpy as np
import faiss
import torch
import psycopg2
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
ALPHA = 0.90
BETA = 0.07
GAMMA = 0.03

TOP_K_CONTEXT = 10

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
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD EMBEDDING MODEL (SciBERT)
# -----------------------------
embed_tokenizer = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased"
)

embed_model = AutoModel.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    use_safetensors=True
).to(device)

embed_model.eval()

# -----------------------------
# LOAD GENERATION MODEL (RAG)
# -----------------------------
gen_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
).to(device)

gen_model.eval()

# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
index = faiss.read_index("faiss_index.bin")
paper_ids = np.load("paper_ids.npy")

print(f"DEBUG: FAISS index size: {index.ntotal}, paper_ids size: {len(paper_ids)}")
if index.ntotal != len(paper_ids):
    print(f"WARNING: MISMATCH! Index has {index.ntotal} papers but paper_ids has {len(paper_ids)}")

# -----------------------------
# LOAD GNN EMBEDDINGS
# -----------------------------
gnn_embeddings = torch.load("gnn_embeddings.pt").numpy()
gnn_paper_ids = np.load("paper_ids_gnn.npy")

print("Loaded GNN embeddings:", gnn_embeddings.shape)
print(f"DEBUG: GNN paper_ids size: {len(gnn_paper_ids)}")

gnn_id_to_index = {str(pid): i for i, pid in enumerate(gnn_paper_ids)}

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_query(text):

    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = embed_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    embedding = embedding.cpu().numpy().astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


# -----------------------------
# FETCH PAPER TEXT
# -----------------------------
def fetch_paper_details(paper_ids):

    cur.execute(
        """
        SELECT paper_id, title, abstract
        FROM papers
        WHERE paper_id = ANY(%s)
        """,
        (paper_ids,)
    )

    rows = cur.fetchall()

    paper_dict = {}

    for pid, title, abstract in rows:

        text = ""

        if title:
            text += title + ". "

        if abstract:
            text += abstract

        paper_dict[str(pid)] = text

    return paper_dict


# -----------------------------
# GENERATE RAG ANSWER
# -----------------------------
def generate_answer(query, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a research assistant.

Using the research paper abstracts below, answer the question clearly.

Question:
{query}

Research Paper Abstracts:
{context_text}

Instructions:
- Summarize key ideas from the papers
- Explain the concept clearly
- Mention important techniques if present

Answer:
"""

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():

        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=200
        )

    answer = gen_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def search(query, top_k=5, generate=False):

    query_vec = embed_query(query)

    scores, indices = index.search(query_vec, 200)

    results = []

    for idx, score in zip(indices[0], scores[0]):

        paper_id = str(paper_ids[idx])
        similarity = float(score)

        results.append((paper_id, similarity))
    
    # DEBUG: Show what indices and paper_ids we got
    if len(results) > 0:
        print(f"DEBUG SEARCH: First index from FAISS: {indices[0][0]}, maps to paper_id: {results[0][0]}")

    paper_id_list = [r[0] for r in results]

    # -----------------------------
    # NORMALIZE SEMANTIC SIMILARITY
    # -----------------------------
    sim_values = [r[1] for r in results]

    max_sim = max(sim_values)
    min_sim = min(sim_values)

    # -----------------------------
    # FETCH PAGERANK
    # -----------------------------
    cur.execute(
        """
        SELECT paper_id, pagerank_score
        FROM papers
        WHERE paper_id = ANY(%s)
        """,
        (paper_id_list,)
    )

    pagerank_dict = {str(k): v for k, v in cur.fetchall()}

    pr_values = list(pagerank_dict.values())

    max_pr = max(pr_values) if pr_values else 1
    min_pr = min(pr_values) if pr_values else 0

    # ----------------------------
    # GENERATE QUERY GNN EMBEDDING
    # ----------------------------
    # Create query representation in GNN space by averaging top semantic results
    query_gnn_embeddings = []
    
    for paper_id, _ in results[:50]:  # Use top 50 semantic matches
        idx = gnn_id_to_index.get(paper_id)
        if idx is not None:
            query_gnn_embeddings.append(gnn_embeddings[idx])
    
    if query_gnn_embeddings:
        query_gnn = np.mean(query_gnn_embeddings, axis=0)
        query_gnn = query_gnn / (np.linalg.norm(query_gnn) + 1e-8)
    else:
        query_gnn = None

    # ----------------------------
    # HYBRID SCORING (Semantic + PageRank + GNN)
    # ----------------------------
    gnn_scores = []
    
    if query_gnn is not None:
        for paper_id, _ in results:
            idx = gnn_id_to_index.get(paper_id)
            if idx is not None:
                gnn_vec = gnn_embeddings[idx]
                sim = np.dot(query_gnn, gnn_vec) / (
                    np.linalg.norm(query_gnn) *
                    np.linalg.norm(gnn_vec) + 1e-8
                )
                gnn_scores.append(sim)
    
    max_gnn = max(gnn_scores) if gnn_scores else 1
    min_gnn = min(gnn_scores) if gnn_scores else 0

    hybrid_results = []

    for paper_id, similarity in results:

        normalized_sim = (
            (similarity - min_sim) /
            (max_sim - min_sim + 1e-8)
        )

        pagerank = pagerank_dict.get(paper_id, 0.0)

        normalized_pr = (
            (pagerank - min_pr) /
            (max_pr - min_pr + 1e-8)
        )

        # Calculate GNN score for this paper
        normalized_gnn = 0.0
        if query_gnn is not None:
            idx = gnn_id_to_index.get(paper_id)
            if idx is not None:
                gnn_vec = gnn_embeddings[idx]
                gnn_score = np.dot(query_gnn, gnn_vec) / (
                    np.linalg.norm(query_gnn) *
                    np.linalg.norm(gnn_vec) + 1e-8
                )
                normalized_gnn = (
                    (gnn_score - min_gnn) /
                    (max_gnn - min_gnn + 1e-8)
                )

        # Hybrid score: semantic + pagerank + GNN
        final_score = (
            ALPHA * normalized_sim +
            BETA * normalized_pr +
            GAMMA * normalized_gnn
        )

        hybrid_results.append(
            (paper_id, final_score)
        )

    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    top_papers = [p[0] for p in hybrid_results[:top_k]]

    # -----------------------------
    # GENERATE RAG ANSWER (optional)
    # -----------------------------
    if generate:

        print("\nTop Papers:\n")

        for i, pid in enumerate(top_papers):
            print(f"{i+1}. {pid}")

        paper_texts = fetch_paper_details(top_papers)

        contexts = [
            paper_texts[p]
            for p in top_papers
            if p in paper_texts
        ]

        print("\nGenerating Research Answer...\n")

        answer = generate_answer(query, contexts)

        print("RAG Answer:\n")
        print(answer)

    return top_papers


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    user_query = input("Enter your query: ")

    search(user_query, generate=True)

    cur.close()
    conn.close()