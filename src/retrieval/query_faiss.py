import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# LOAD MODEL (SciBERT)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased",
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
# SEARCH FUNCTION
# -----------------------------
def search(query, top_k=5):
    query_vec = embed_query(query)
    scores, indices = index.search(query_vec, top_k)

    print("\nTop Results:\n")
    for i in range(top_k):
        print("Paper ID:", paper_ids[indices[0][i]], 
              "Similarity:", round(float(scores[0][i]), 4))


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    search(user_query)