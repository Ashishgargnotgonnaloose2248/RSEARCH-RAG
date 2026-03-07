import psycopg2
import json

QUERIES = [
"graph neural networks",
"deep learning healthcare",
"convolutional neural networks image classification",
"federated learning systems",
"transformer models natural language processing",
"emotion recognition deep learning",
"machine learning fairness",
"neural recommendation systems",
"knowledge graph embedding",
"sequence to sequence models",
"reinforcement learning robotics",
"medical image segmentation",
"object detection computer vision",
"graph representation learning",
"self supervised learning",
"large language models",
"text summarization neural networks",
"named entity recognition",
"speech recognition deep learning",
"multimodal learning",
"federated recommendation systems",
"graph attention networks",
"semantic search embeddings",
"contrastive learning representations",
"visual question answering",
"recommendation systems collaborative filtering",
"deep reinforcement learning",
"domain adaptation machine learning",
"medical diagnosis machine learning",
"neural architecture search",
"time series forecasting deep learning",
"video understanding neural networks",
"autonomous driving perception",
"image generation diffusion models",
"style transfer neural networks",
"document retrieval semantic search",
"citation network analysis",
"fraud detection machine learning",
"anomaly detection neural networks",
"topic modeling text mining",
"sentiment analysis transformers",
"recommendation systems deep learning",
"cross modal retrieval",
"knowledge distillation neural networks",
"few shot learning",
"meta learning algorithms",
"explainable artificial intelligence",
"graph clustering algorithms",
"neural ranking models",
"information retrieval neural networks"
]

conn = psycopg2.connect(
    dbname="RagDb",
    user="postgres",
    password="Ashishgarg22#",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

queries_json = {}

for query in QUERIES:

    sql = """
    SELECT paper_id
    FROM papers
    WHERE title ILIKE %s
       OR abstract ILIKE %s
    LIMIT 10
    """

    keyword = "%" + query.split()[0] + "%"

    cur.execute(sql, (keyword, keyword))
    rows = cur.fetchall()

    paper_ids = [r[0] for r in rows]

    queries_json[query] = paper_ids


with open("src/evaluation/queries.json","w") as f:
    json.dump(queries_json, f, indent=2)

print("Created queries.json with", len(queries_json), "queries")

cur.close()
conn.close()