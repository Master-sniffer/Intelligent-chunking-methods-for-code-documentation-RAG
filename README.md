# Retrieval Pipeline Using Open-Source Embedding Models

This document presents an end-to-end overview of a retrieval pipeline built with open-source embedding models. The goal is to evaluate how well different models and methods retrieve relevant text chunks based on user queries, using several datasets and evaluation metrics.

---

## Dataset Overview

We used three datasets from the [chunking_evaluation repository](https://github.com/brandonstarxel/chunking_evaluation/tree/main/chunking_evaluation/evaluation_framework/general_evaluation_data), each offering different types of content and queries:

- **Chatlogs**: Real conversation logs with queries and corresponding excerpts.  
- **State of the Union**: Political speeches paired with related questions.  
- **Wikitext**: Wikipedia article snippets matched to relevant queries.

Each dataset includes:
- A document collection (corpus)
- A list of questions (`questions_df.csv`)
- The relevant answers or excerpts for each query

These datasets are automatically downloaded and prepared by the pipeline for evaluation.

---

## Tools and Libraries Used

### Embedding Models
- `sentence-transformers`  
- Models like:
  - `all-MiniLM-L6-v2` – Lightweight, fast
  - `multi-qa-mpnet-base-dot-v1` – Higher accuracy, heavier

### Text Processing & Tokenization
- `tiktoken` – For token-based chunking
- `rank_bm25` – Traditional sparse retrieval

### GPU Acceleration
- `PyTorch` – Enables GPU-backed embedding generation

---

## Evaluation Metrics

I assessed how well the retrieval works using the following metrics:

- **Precision**: What fraction of the retrieved tokens are relevant  
- **Recall**: What fraction of the relevant tokens are retrieved  
- **F1 Score**: The harmonic mean of precision and recall  
- **Semantic Similarity**: Measures how semantically close the retrieved and target texts are, using embeddings and cosine similarity

The precision and recall are token-based, while semantic similarity uses sentence embeddings.

---

## Methodology

My approach includes several core components:

### 1. Chunking Text  
We use a `FixedTokenChunker` to split text into fixed-size token chunks with optional overlaps. This helps manage long documents and align better with transformer model limits.

### 2. Embedding Text  
An `EmbeddingFunction` class wraps around the sentence-transformers framework. If a GPU is available, the model loads on CUDA for faster computation.

### 3. Retrievers
We explore two types of retrievers:

- **HybridRetriever**: Combines sparse (BM25) and dense (embedding) methods  
- **CrossEncoderRetriever**: Uses cross-encoders for more precise scoring, though at higher computational cost

### 4. Evaluation Pipeline
The process:
- Chunks the corpus
- Embeds text
- Retrieves relevant chunks for each query
- Calculates metrics (precision, recall, etc.)
- Uses parallelization to speed things up

### 5. Experimentation (Grid Search)
We systematically test different parameters:
- Chunk sizes (200 vs. 400 tokens)
- Overlap sizes
- Number of chunks to retrieve
- Model types
- Retriever methods

---

## Key Findings

Here are the most important observations:

1. **Chunk Size Trade-offs**  
   - Larger chunks improve recall but reduce precision  
   - Smaller chunks increase precision but risk missing content  

2. **More Retrieved Chunks = Better Recall**  
   - Pulling more chunks improves recall but often lowers precision  
   - There’s a trade-off between completeness and focus  

3. **Model Performance**  
   - `multi-qa-mpnet-base-dot-v1` generally performs better but is slower  
   - `all-MiniLM-L6-v2` is faster and still solid for most cases  

4. **Retriever Differences**  
   - Cross-encoders deliver higher precision  
   - Hybrid retrievers offer better balance between speed and performance  

5. **What Matters Most?**
   - Chunk size affects results the most  
   - Model choice is next in importance  
   - Number of chunks retrieved also influences outcomes  

Visualizations generated include:
- Precision-recall curves  
- F1 score comparisons  
- Parameter importance charts  
- Chunk size vs metric plots  

---

## Challenges and Workarounds

| Challenge | Solution |
|----------|----------|
| Large text corpora consume a lot of memory | Batched processing and efficient data structures |
| Hard to optimize both precision and recall | Combined dense and sparse methods in `HybridRetriever` |
| Evaluation was slow | Parallelized evaluation using `ThreadPoolExecutor` |
| GPU usage wasn’t always optimal | Dynamically selected CUDA if available; used batch processing |

---

## How to Reproduce the Results

Run it from the beginning to install all the dependencies or manually:

```python
pip install -q rank_bm25 kaleido numpy pandas 
```

### 3. Analyze Results

Check the `visualizations` folder for:
- `f1_comparison.png`
- `parameter_importance.png`
- `precision_recall_tradeoff.png`
- `chunk_size_vs_metrics.png`

CSV files with full results are in the `results/` directory.

### 4. Customize and Extend

You can easily:
- Switch to another dataset
- Adjust chunk sizes and overlaps
- Try new embedding models
- Create custom retrievers by extending base classes
- Add your own metrics
