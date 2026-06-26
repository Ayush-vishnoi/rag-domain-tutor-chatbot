# Chapter 7: Retrieval-Augmented Generation (RAG)

## 7.1 Why RAG?

Large Language Models have two core limitations:
1. **Knowledge cutoff:** They only know information from their training data
2. **Hallucination:** They can generate plausible but factually incorrect text

RAG addresses both by grounding the model's responses in retrieved documents at inference time, without retraining.

---

## 7.2 RAG Pipeline

A standard RAG pipeline has five stages:

### Stage 1: Document Loading
Load raw documents (PDFs, markdown, web pages) using document loaders.

### Stage 2: Chunking
Split documents into smaller chunks (typically 512–1024 tokens) with overlap (100–200 tokens) to preserve context at boundaries.

**Chunking strategies:**
- Fixed-size chunking
- Recursive character splitting (respects sentence/paragraph boundaries)
- Semantic chunking (splits based on embedding similarity)

### Stage 3: Embedding
Convert each chunk into a dense vector using an embedding model (e.g., `text-embedding-ada-002`, `sentence-transformers/all-MiniLM-L6-v2`).

### Stage 4: Retrieval
Given a user query, embed the query and perform approximate nearest-neighbor search in the vector store to retrieve top-k relevant chunks.

**Retrieval types:**
- Similarity search (cosine/dot product)
- MMR (Maximal Marginal Relevance) — balances relevance and diversity
- Hybrid search — combines dense retrieval with BM25 keyword search

### Stage 5: Generation
Pass the retrieved chunks as context in the prompt to the LLM, which generates a grounded answer.

```
Prompt = System instructions + Retrieved context + User question
```

---

## 7.3 Advanced RAG Techniques

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the raw query, ask the LLM to generate a hypothetical answer, then embed that. This bridges the lexical gap between short queries and long documents.

### Re-ranking
After initial retrieval, use a cross-encoder re-ranker to re-score and reorder results. Cross-encoders are slower but more accurate than bi-encoders.

### Query Expansion
Generate multiple query variants and retrieve for each, then merge results.

### Parent-Child Chunking
Store small chunks for retrieval precision but return larger parent chunks for generation context.

---

## 7.4 Hallucination in RAG

Even with retrieved context, LLMs can hallucinate. Types:

1. **Intrinsic hallucination:** Output contradicts the retrieved context
2. **Extrinsic hallucination:** Output contains information not present in the context (but may still be true)

### Hallucination Detection Methods

**1. NLI-based (Natural Language Inference)**
Use an NLI model to check if each sentence in the answer is entailed by the retrieved context.
- Entailed → Faithful
- Contradicted → Hallucination
- Neutral → Unverifiable

**2. Self-consistency**
Sample multiple outputs with temperature > 0. If answers are inconsistent, the model is uncertain (likely hallucinating).

**3. RAGAS (RAG Assessment)**
A framework with automated metrics:
- **Faithfulness:** Is the answer supported by the context?
- **Answer Relevancy:** Is the answer relevant to the question?
- **Context Precision:** Are the retrieved chunks relevant?
- **Context Recall:** Are all necessary chunks retrieved?

**4. Semantic similarity**
Compute cosine similarity between the answer and retrieved context embeddings. Low similarity indicates potential hallucination.

---

## 7.5 Evaluation Framework

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| Faithfulness | Answer supported by context | NLI entailment score |
| Answer Relevancy | Answer addresses the question | Embed answer & question, compute similarity |
| Context Precision | Retrieved docs are relevant | Proportion of relevant docs in top-k |
| Context Recall | All relevant docs retrieved | Coverage of ground truth |

---

## 7.6 Vector Databases

| Database | Type | Notes |
|----------|------|-------|
| FAISS | In-memory/local | Fast, no server needed, from Meta |
| Chroma | Local/server | Easy to use, persistent |
| Pinecone | Cloud | Managed, scalable |
| Weaviate | Cloud/self-hosted | Supports hybrid search |

---

## 7.7 Summary

- RAG grounds LLM responses in retrieved documents, reducing hallucinations
- The pipeline: Load → Chunk → Embed → Retrieve → Generate
- Hallucination can be detected via NLI, self-consistency, or RAGAS metrics
- Advanced techniques (HyDE, re-ranking, hybrid search) improve retrieval quality
