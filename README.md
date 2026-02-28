# 📚 RESEARCH-GNN-RAG

**Graph-Enhanced Scholarly Retrieval, Citation Intelligence & RAG System**

---

## 🎯 Overview

**RESEARCH-GNN-RAG** is a research-grade academic intelligence system that integrates:

- 🔍 **Semantic Retrieval**
- 🕸️ **Citation Graph Modeling**
- 🧠 **Graph Neural Networks** (GCN, GraphSAGE, Spatio-Temporal)
- 📊 **Citation Prediction & Forecasting**
- ⭐ **Query-Based Ranking**
- 🤖 **Retrieval-Augmented Generation (RAG)**

The system fetches Computer Science research papers from the Semantic Scholar API, constructs a validated citation network, learns structural representations using GNNs, enhances ranking accuracy, and generates citation-grounded research responses.

---

## 🧠 Problem Statement

Traditional research paper search engines rely primarily on:

- Keyword matching
- Raw citation count sorting

These approaches:

- Ignore citation network structure
- Do not model influence propagation
- Fail to account for temporal citation dynamics
- Provide limited ranking intelligence

**This project proposes:**  
A graph-enhanced ranking system that integrates **semantic similarity**, **citation structure**, and **temporal modeling** to improve scholarly retrieval accuracy and reliability.

---

## 🏗️ Complete System Architecture

```
Semantic Scholar API
        ↓
PostgreSQL Storage
        ↓
Citation Preprocessing & Validation
        ↓
Graph Construction
        ↓
Feature Engineering
        ↓
FFN Baseline → GCN → GraphSAGE → Spatio-Temporal GNN
        ↓
Multi-Task Learning Heads
        ↓
FAISS Semantic Retrieval
        ↓
Ranking Fusion
        ↓
RAG Answer Generation
```

---

## 🔹 Stage 1 — Data Acquisition

| Property | Detail |
|---|---|
| **Source** | Semantic Scholar Graph API |
| **Domain** | Computer Science |
| **Time Range** | Last 2–3 years |
| **Dataset Size** | 1,000–1,500 papers |

**Extracted Fields:**
- `paper_id`, `title`, `abstract`, `year`
- `citationCount` (global citation count)
- `references` (citation edges)

All metadata is stored in **PostgreSQL**.

---

## 🔹 Stage 2 — Citation Preprocessing & Validation

**Data Cleaning:**
- Remove duplicate paper IDs
- Remove papers with missing abstracts
- Strict domain filtering (Computer Science only)
- Keep only internal citation edges
- Validate graph connectivity
- Normalize citation counts

### 📌 Citation Metrics (Important Distinction)

| Metric | Meaning |
|---|---|
| **Global Citation Count** | Total citations across entire Semantic Scholar |
| **Internal In-Degree** | Citations received within dataset |
| **Internal Out-Degree** | References made within dataset |

> ⚠️ These metrics are **never mixed**.

### 📌 Citation Normalization

To reduce age bias:

$$\text{NormalizedCitation} = \frac{\text{citationCount}}{(\text{CurrentYear} - \text{Year} + 1)}$$

or

$$\log(1 + \text{citationCount})$$

---

## 🔹 Stage 3 — Graph Construction

We construct a **directed citation graph**:

- **Nodes** → Research papers
- **Edges** → Citation relationships (Paper A → Paper B)

Converted into **PyTorch Geometric** format:
- `x` → Node feature matrix
- `edge_index` → Citation edge tensor

---

## 🔹 Stage 4 — Feature Engineering

Each node includes:

| Feature | Dimension |
|---|---|
| SciBERT embedding | 768-dim semantic vector |
| Normalized citation count | Scalar |
| Publication year (normalized) | Scalar |
| Graph statistics (optional) | Degree centrality |

Produces feature matrix: **X ∈ ℝ^(N×F)**

---

## � Stage 5 — Graph Neural Network Design

Progressive model development strategy:

### 5.1 Feed Forward Neural Network (Baseline)
- Uses node features only, no graph structure
- **Purpose:** Establish semantic-only performance baseline

### 5.2 Graph Convolutional Network (GCN)
- Aggregates neighbor information with uniform weighting
- **Purpose:** Introduce citation-aware learning

### 5.3 GraphSAGE *(Primary Backbone)*
- Learnable neighborhood aggregation
- Scalable and expressive
- Handles sparse citation graphs effectively

### 5.4 Spatio-Temporal Graph Modeling
- Time decay weighting
- Citation growth modeling
- Year-based normalization

> Captures: **Spatial structure** (citation network) + **Temporal evolution** (publication trends)

---

## 🔹 Stage 6 — Multi-Task Learning

The shared GNN backbone supports multiple tasks:

| Task | Method | Metric |
|---|---|---|
| 🏷️ Paper Classification | Predict CS subfield | Accuracy |
| 🔗 Citation Prediction | Edge existence via embedding similarity + MLP, with negative sampling | AUC |
| 🔮 Future Citation Forecasting | Regression head with log transformation | RMSE |

---

## � Stage 7 — Semantic Retrieval & Ranking Fusion

**Step 1 — Semantic Retrieval**
```
User Query → SciBERT Embedding → FAISS → Top 100 Candidates
```

**Step 2 — Graph-Based Re-Ranking**

$$\text{FinalScore} = \alpha \cdot \text{SemanticSimilarity} + \beta \cdot \text{GraphImportance} + \gamma \cdot \text{TemporalWeight}$$

Returns **Top-K ranked papers**.

---

## � Stage 8 — Retrieval-Augmented Generation (RAG)

Top-ranked papers are sent to an LLM which:
- Uses titles and abstracts as context
- Generates **citation-grounded responses**
- Produces structured scholarly answers

> RAG acts as the **final presentation layer**.

---

## � Evaluation Strategy

| Task | Metric |
|---|---|
| Ranking | Precision@10, Recall@10, nDCG@10 |
| Citation Prediction | AUC |
| Forecasting | RMSE, MAE |
| Classification | Accuracy |

---

## 📈 Graph Validation Criteria

Before GNN training, we validate:

| Criterion | Minimum Requirement |
|---|---|
| Average degree | ≥ 3 |
| Largest connected component | ≥ 60% of total nodes |

Also checked: number of nodes/edges, degree distribution.

---

## 🗄️ Database Schema

### `papers`

| Column | Description |
|---|---|
| `paper_id` | Primary key |
| `title` | Paper title |
| `abstract` | Abstract text |
| `year` | Publication year |
| `citation_count` | Global citation count |
| `reference_count` | Reference count |

### `citations`

| Column | Description |
|---|---|
| `citing_paper_id` | Source paper |
| `cited_paper_id` | Target paper |

> Composite primary key `(citing_paper_id, cited_paper_id)` prevents duplicate edges.

---

## � Vector Database — FAISS

| Reason | Detail |
|---|---|
| Speed | Fast nearest neighbor search |
| Weight | Lightweight, no server required |
| Scale | Suitable for 1,000–1,500 vectors |
| Integration | Seamless with PyTorch |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Data Source | Semantic Scholar API |
| Database | PostgreSQL |
| Vector Search | FAISS |
| Embeddings | SciBERT |
| Graph Learning | PyTorch Geometric |
| Deep Learning | PyTorch |
| LLM | GPT / Gemini |
| Language | Python |

---

## 📂 Project Structure

```
RSEARCH-GNN-RAG/
├── src/
│   ├── api/            # Semantic Scholar API fetchers
│   ├── db/             # PostgreSQL schema, init, storage
│   ├── graph/          # Graph construction & validation
│   ├── models/         # FFN, GCN, GraphSAGE, Spatio-Temporal GNN
│   ├── embeddings/     # SciBERT embeddings & FAISS indexing
│   ├── evaluation/     # Metrics & evaluation scripts
│   └── rag/            # RAG pipeline & LLM integration
├── requirements.txt
├── README.md
└── .env                # API keys & DB credentials (not committed)
```

---

## 🔐 Security

- API keys stored in `.env`
- `.env` excluded from Git via `.gitignore`
- No credentials committed to the repository

---

## � Research Contributions

This project contributes:

1. **Graph-enhanced scholarly ranking** using citation network topology
2. **Multi-task citation intelligence** modeling (classification + prediction + forecasting)
3. **Temporal citation analysis** with time-decay modeling
4. **Fusion of semantic and structural signals** for ranking
5. **Citation-grounded RAG** for structured research answer generation

---

## � Final Statement

**RESEARCH-GNN-RAG** is a modular, research-focused, graph-centric scholarly intelligence system.

It integrates:
- Semantic Retrieval → Citation Graph Learning → Temporal Modeling
- Multi-Task Neural Learning → Ranking Fusion → Retrieval-Augmented Generation

...to move beyond traditional academic search systems.

---

## 👤 Author

**Anurag Mishra**  
[GitHub → github.com/anuragmishra5159](https://github.com/anuragmishra5159)

**Ashish Garg**
[GitHub → github.com/Ashishgargnotgonnaloose2248](https://github.com/Ashishgargnotgonnaloose2248)



---

## 📄 License

This project is licensed under the [MIT License](LICENSE).