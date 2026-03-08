import json
import numpy as np
import io
import contextlib
import warnings
import logging
import os

# -----------------------------
# SILENCE WARNINGS / LOGS
# -----------------------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from src.retrieval.query_faiss import search

K = 5
SEARCH_K = 50  # retrieve deeper list for better evaluation


# -----------------------------
# RUN SEARCH WITHOUT PRINTING
# -----------------------------
def silent_search(query):

    f = io.StringIO()

    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        results = search(query, top_k=SEARCH_K)

    return [str(r) for r in results]


# -----------------------------
# METRICS
# -----------------------------
def precision_at_k(results, relevant):

    retrieved = results[:K]
    relevant = [str(r) for r in relevant]

    return len(set(retrieved) & set(relevant)) / K


def recall_at_k(results, relevant):

    retrieved = results[:K]
    relevant = [str(r) for r in relevant]

    if len(relevant) == 0:
        return 0

    return len(set(retrieved) & set(relevant)) / len(relevant)


def reciprocal_rank(results, relevant):

    relevant = [str(r) for r in relevant]

    for i, r in enumerate(results):
        if r in relevant:
            return 1 / (i + 1)

    return 0


# -----------------------------
# LOAD QUERIES
# -----------------------------
with open("src/evaluation/queries.json") as f:
    queries = json.load(f)


# -----------------------------
# OPTIONAL: CHECK ID COVERAGE
# -----------------------------
try:
    paper_ids = np.load("paper_ids.npy", allow_pickle=True)
    paper_ids = set(str(p) for p in paper_ids)

    total = 0
    missing = 0

    for q, rel in queries.items():
        for r in rel:
            total += 1
            if str(r) not in paper_ids:
                missing += 1

    print("\nGround truth coverage in FAISS index:")
    print("Missing relevant papers:", missing, "/", total)

except:
    print("paper_ids.npy not found — skipping index coverage check")


# -----------------------------
# RUN EVALUATION
# -----------------------------
precision_scores = []
recall_scores = []
mrr_scores = []

print("\nRunning evaluation on", len(queries), "queries...\n")

# DEBUG: Test first query
test_query = list(queries.items())[0]
print(f"DEBUG - First query: '{test_query[0]}'")
print(f"DEBUG - Ground truth papers: {test_query[1][:3]}")  # First 3
test_results = silent_search(test_query[0])
print(f"DEBUG - Top 5 results: {test_results[:5]}")
print(f"DEBUG - Any match? {set(test_results[:5]) & set(str(r) for r in test_query[1])}\n")

for query, relevant in queries.items():

    results = silent_search(query)

    precision_scores.append(precision_at_k(results, relevant))
    recall_scores.append(recall_at_k(results, relevant))
    mrr_scores.append(reciprocal_rank(results, relevant))


# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\nEvaluation Results\n")

print("Precision@5:", round(np.mean(precision_scores), 4))
print("Recall@5:", round(np.mean(recall_scores), 4))
print("MRR:", round(np.mean(mrr_scores), 4))