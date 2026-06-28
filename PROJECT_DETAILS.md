# 📘 Project Deep Dive — RAG-Based NLP Domain Tutor Chatbot

Yeh file project ke andar ki poori detail explain karti hai — data kahan se aaya, kaise process hota hai, aur system end-to-end kaise kaam karta hai.

---

## 📂 Data Sources — Corpus Kahan Se Aaya?

Saara knowledge `syllabus_docs/` folder mein `.md` files ke form mein hai. Yeh documents manually create kiye gaye hain, real academic references ke basis par:

| File | Kya cover karta hai | Reference |
|------|---------------------|-----------|
| `nlp_course_syllabus.md` | CS 5340 course structure — 8 modules, assignments, grading policy | Standard university NLP curriculum |
| `chapter1_foundations.md` | Tokenization, stemming, BoW, TF-IDF, N-gram LM, BLEU, ROUGE | Jurafsky & Martin — "Speech and Language Processing" |
| `chapter2_classical_nlp.md` | HMM, POS tagging, NER (BIO scheme), CRF, Naive Bayes, parsing | Manning & Schütze — "Foundations of Statistical NLP" |
| `chapter3_embeddings.md` | Word2Vec (Skip-gram/CBOW), GloVe, FastText, static vs contextual | Mikolov et al. (2013), Pennington et al. (2014) |
| `chapter4_sequence_models.md` | RNN, vanishing gradients, LSTM, GRU, Bidirectional RNN, Seq2Seq + Attention | Hochreiter & Schmidhuber (1997), Bahdanau et al. (2015) |
| `chapter5_transformers.md` | Attention mechanism, Transformer architecture, BERT, GPT family | Vaswani et al. (2017) "Attention is All You Need", Devlin et al. (2018) |
| `chapter6_llms.md` | Scaling laws, emergent abilities, RLHF, prompt engineering, hallucination | Kaplan et al. (2020), Ouyang et al. (2022) InstructGPT |
| `chapter7_rag.md` | RAG pipeline, hallucination types, RAGAS metrics, vector databases | Lewis et al. (2020), RAGAS framework |

> Koi bhi naya `.md` file `syllabus_docs/` mein daalo — pipeline automatically usse load, chunk, aur embed kar lega next run par.

---

## ⚙️ System Poora Kaise Kaam Karta Hai — Step by Step

### Step 1 — Document Loading
```
syllabus_docs/*.md
       ↓
TextLoader (LangChain) reads each file as plain text
       ↓
metadata['source'] = filename  ← taaki pata chale answer kahan se aaya
       ↓
List of Document objects
```

### Step 2 — Chunking
Bade documents ko chhote pieces mein todo, overlapping window ke saath:
```
RecursiveCharacterTextSplitter(
    chunk_size    = 800 tokens,
    chunk_overlap = 150 tokens   ← boundary par context preserve karne ke liye
)
```
Ek 5000-word chapter roughly 8–10 chunks banega.

### Step 3 — Embedding
Har chunk ko ek dense vector mein convert karo:
```
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
- Local model — koi API cost nahi
- 384-dimensional vectors produce karta hai
- Semantically similar text → similar vectors

### Step 4 — Vector Store (FAISS)
Saare chunk vectors ek in-memory FAISS index mein store hote hain:
```
FAISS.from_documents(chunks, embeddings)
```
- Meta se banaya gaya fast similarity search library
- In-memory — koi database setup nahi chahiye

### Step 5 — Retrieval (MMR)
User ka question aata hai → embed karo → FAISS mein similar chunks dhundo:
```
MMR Retriever (Maximal Marginal Relevance)
    fetch_k = 10 candidates laao
    k = 4 final chunks return karo
```
MMR sirf top-4 similar chunks nahi laata — diversity bhi ensure karta hai taaki same content repeat na ho.

### Step 6 — Generation (Groq LLM)
Retrieved 4 chunks + user question ek prompt mein pack hota hai:
```
System: Tu ek NLP tutor hai, course materials use kar jawab dene ke liye
Context: [4 retrieved chunks]
Question: [user ka question]
       ↓
Groq API → llama-3.3-70b-versatile
       ↓
Answer
```
Groq free tier pe extremely fast inference deta hai (usually < 2 seconds).

### Step 7 — Hallucination Detection
Answer kitna grounded hai context mein — cosine similarity se measure karo:
```
answer_embedding  = embed(answer)
context_embedding = embed(joined retrieved chunks)
faithfulness_score = cosine_similarity(answer_emb, context_emb)

score >= 0.75 → ✅ Grounded
score <  0.75 → ⚠️ Potential hallucination
```

---

## 🔁 Complete Flow Diagram

```
User Question
     │
     ▼
Embed Question ──────────────────────────────────────────┐
     │                                                   │
     ▼                                                   │
FAISS Vector Store                                       │
(all syllabus_docs chunks)                               │
     │                                                   │
     ▼                                                   │
MMR Retrieval → Top 4 relevant chunks                    │
     │                                                   │
     ▼                                                   │
Build Prompt                                             │
[Tutor instruction + Context + Question]                 │
     │                                                   │
     ▼                                                   │
Groq LLM (llama-3.3-70b-versatile)                      │
     │                                                   │
     ▼                                                   │
Answer ──────────────────────────────────────────────────┘
     │                        │
     ▼                        ▼
Display to User      Hallucination Score
                     (cosine sim: answer vs context)
```

---

## 🧠 Model & Tool Choices — Kyun Yeh?

| Component | Tool | Kyun choose kiya |
|-----------|------|-----------------|
| LLM | Groq — `llama-3.3-70b-versatile` | Free, fast (< 2s), 70B model intelligent enough for complex NLP explanations |
| Embeddings | `all-MiniLM-L6-v2` | Completely local — zero API cost, good semantic quality for retrieval |
| Vector Store | FAISS | No server needed, in-memory, perfect for small-medium corpus |
| RAG Framework | LangChain LCEL | Clean chain composition, modular, easy to swap components |
| UI | Streamlit | Rapid deployment, built-in secrets management, free cloud hosting |
| Retrieval | MMR | Better than pure similarity — avoids returning 4 near-identical chunks |

---

## 📊 Hallucination Detection — Kaise Kaam Karta Hai?

Yeh project ek simple but effective **semantic faithfulness** score use karta hai:

```python
context = " ".join([doc.page_content for doc in source_docs])
score = cosine_similarity(embed(answer), embed(context))
```

Intuition: Agar answer retrieved context ke close hai embedding space mein, toh answer grounded hai. Agar bahut door hai, model apni training se kuch bana raha hai.

**Limitation:** Yeh NLI-based detection se simpler hai — false positives ho sakte hain. Production mein RAGAS ya NLI model use karna better hoga.

---

## 🗂️ Corpus Coverage Map

```
Module 1 — Foundations          ✅ chapter1_foundations.md
Module 2 — Classical NLP        ✅ chapter2_classical_nlp.md
Module 3 — Word Embeddings      ✅ chapter3_embeddings.md
Module 4 — Sequence Models      ✅ chapter4_sequence_models.md
Module 5 — Transformers         ✅ chapter5_transformers.md
Module 6 — LLMs                 ✅ chapter6_llms.md
Module 7 — RAG                  ✅ chapter7_rag.md
Module 8 — NLP Applications     ⚠️  Only covered in syllabus overview
Course Structure / Assignments  ✅ nlp_course_syllabus.md
```

---

## 🔧 Agar Apna Corpus Add Karna Ho

1. Apna `.md` file `syllabus_docs/` mein daalo
2. App restart karo (ya Streamlit Cloud pe redeploy karo)
3. `load_vectorstore()` automatically naya file pick kar lega — koi code change nahi

PDF support add karna ho toh `rag_pipeline.py` mein `TextLoader` ko `PyPDFLoader` se replace karo:
```python
from langchain_community.document_loaders import PyPDFLoader
```
