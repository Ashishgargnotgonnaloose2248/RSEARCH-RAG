import json
import numpy as np
import io
import contextlib
import warnings
import logging
import os

# suppress warnings and library logs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from src.retrieval.query_faiss import search

K = 5


def silent_search(query):
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        results = search(query, top_k=10)
    return results


def precision_at_k(results, relevant):
    retrieved = results[:K]
    return len(set(retrieved) & set(relevant)) / K


def recall_at_k(results, relevant):
    retrieved = results[:K]
    return len(set(retrieved) & set(relevant)) / len(relevant)


def reciprocal_rank(results, relevant):
    for i, r in enumerate(results):
        if r in relevant:
            return 1 / (i + 1)
    return 0


with open("src/evaluation/queries.json") as f:
    queries = json.load(f)

precision_scores = []
recall_scores = []
mrr_scores = []

for query, relevant in queries.items():

    results = [r[0] for r in silent_search(query)]

    precision_scores.append(precision_at_k(results, relevant))
    recall_scores.append(recall_at_k(results, relevant))
    mrr_scores.append(reciprocal_rank(results, relevant))


print("\nEvaluation Results with GNN Enabled\n")

print("Precision@5:", round(np.mean(precision_scores), 4))
print("Recall@5:", round(np.mean(recall_scores), 4))
print("MRR:", round(np.mean(mrr_scores), 4))