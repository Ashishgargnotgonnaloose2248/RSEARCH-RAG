import torch
from transformers import AutoTokenizer, AutoModel
import psycopg2
from tqdm import tqdm
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load SciBERT
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    use_safetensors=True
    )
model.to(device)
model.eval()

# DB connection
conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Add embedding column if not exists
cur.execute("""
ALTER TABLE papers
ADD COLUMN IF NOT EXISTS embedding FLOAT8[];
""")
conn.commit()

# Fetch papers
cur.execute("""
SELECT paper_id, abstract
FROM papers
WHERE abstract IS NOT NULL;
""")

rows = cur.fetchall()
print("Total papers to embed:", len(rows))

batch_size = 8  # Safe for RTX 2050 4GB

for i in tqdm(range(0, len(rows), batch_size)):

    batch = rows[i:i+batch_size]
    paper_ids = [r[0] for r in batch]
    abstracts = [r[1] for r in batch]

    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    cls_embeddings = cls_embeddings.cpu().numpy()

    for pid, emb in zip(paper_ids, cls_embeddings):
        cur.execute("""
            UPDATE papers
            SET embedding = %s
            WHERE paper_id = %s;
        """, (emb.tolist(), pid))

    conn.commit()

cur.close()
conn.close()

print("Embedding generation completed.")