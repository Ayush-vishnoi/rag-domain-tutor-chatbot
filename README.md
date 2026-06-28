# 🎓 RAG-Based Domain Tutor Chatbot

A Retrieval-Augmented Generation (RAG) chatbot grounded on an NLP course syllabus and textbook corpus, with a hallucination detection evaluation pipeline.

## 🚀 Live Demo
> Deploy on Streamlit Cloud — [share.streamlit.io](https://rag-domain-tutor-chatbot-aitmpmkqisdtv9gvavpmc7.streamlit.app))

---

## 📁 Project Structure

```
tutor_app/
├── syllabus_docs/                  # Domain knowledge corpus
│   ├── nlp_course_syllabus.md      # NLP course syllabus (modules, assignments, grading)
│   ├── chapter5_transformers.md    # Textbook chapter on Transformers, BERT, GPT
│   └── chapter7_rag.md             # Textbook chapter on RAG + hallucination theory
├── .streamlit/
│   └── secrets.toml                # Local API keys (not pushed to GitHub)
├── app.py                          # Streamlit UI
├── rag_pipeline.py                 # Core RAG + hallucination detection logic
├── tutor_chatbot.ipynb             # Step-by-step documentation notebook
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## 🧠 How It Works

### RAG Pipeline
```
syllabus_docs/ (MD files)
      ↓
TextLoader → RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
      ↓
HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Vector Store
      ↓
User Question → MMR Retriever (top 4 chunks)
      ↓
Tutor Prompt + Retrieved Context → Groq LLM (llama-3.1-8b-instant)
      ↓
Answer + Hallucination Score
```

### Hallucination Detection
Semantic faithfulness score — cosine similarity between answer embedding and retrieved context embedding.
- Score ≥ 0.75 → ✅ Answer is grounded
- Score < 0.75 → ⚠️ Potential hallucination

---

## 📚 Corpus / Document Sources

The `syllabus_docs/` folder contains the domain knowledge. Documents were created based on:

| File | Content | Reference |
|------|---------|-----------|
| `nlp_course_syllabus.md` | CS 5340 course structure — 8 modules, assignments, grading policy | Standard university NLP curriculum |
| `chapter1_foundations.md` | Tokenization, stemming, BoW, TF-IDF, N-gram LM, BLEU, ROUGE | Jurafsky & Martin — "Speech and Language Processing" |
| `chapter2_classical_nlp.md` | HMM, POS tagging, NER (BIO scheme), CRF, Naive Bayes, parsing | Manning & Schütze — "Foundations of Statistical NLP" |
| `chapter3_embeddings.md` | Word2Vec (Skip-gram/CBOW), GloVe, FastText, static vs contextual | Mikolov et al. (2013), Pennington et al. (2014) |
| `chapter4_sequence_models.md` | RNN, vanishing gradients, LSTM, GRU, Bidirectional RNN, Seq2Seq + Attention | Hochreiter & Schmidhuber (1997), Bahdanau et al. (2015) |
| `chapter5_transformers.md` | Attention mechanism, Transformer architecture, BERT, GPT family | Vaswani et al. (2017) "Attention is All You Need", Devlin et al. (2018) |
| `chapter6_llms.md` | Scaling laws, emergent abilities, RLHF, prompt engineering, hallucination | Kaplan et al. (2020), Ouyang et al. (2022) InstructGPT |
| `chapter7_rag.md` | RAG pipeline, hallucination types, RAGAS metrics, vector databases | Lewis et al. (2020), RAGAS framework |

> **To add your own documents:** Add `.md` files to `syllabus_docs/` — the pipeline will automatically load, chunk, and embed them on next run.

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq API — `llama-3.1-8b-instant` (free) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | FAISS (local, in-memory) |
| RAG Framework | LangChain (LCEL) |
| UI | Streamlit |
| Hallucination Detection | Cosine similarity (semantic faithfulness) |

---

## 🛠️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Ayush-vishnoi/rag-domain-tutor-chatbot
cd tutor_app
```

### 2. Create and activate environment
```bash
conda create -n tutor_rag python=3.10 -y
conda activate tutor_rag
pip install -r requirements.txt
```

### 3. Add API keys

Get a free Groq API key from [console.groq.com](https://console.groq.com)

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📓 Jupyter Notebook

`tutor_chatbot.ipynb` contains the full pipeline step by step for documentation and experimentation.

To run the notebook, create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```

Then open the notebook and select `tutor_rag` kernel.

---

## 📦 Dependencies

```
langchain==1.3.11
langchain-community==0.4.2
langchain-core==1.4.8
langchain-huggingface==1.2.2
langchain-groq==1.1.3
langchain-text-splitters==1.1.2
faiss-cpu==1.14.3
sentence-transformers==5.6.0
streamlit==1.58.0
python-dotenv==1.2.2
numpy==2.2.6
pandas==2.3.3
```

---

## 🔑 Key Concepts Demonstrated

- **RAG (Retrieval-Augmented Generation)** — grounding LLM responses in a document corpus
- **LCEL (LangChain Expression Language)** — modern chain composition
- **MMR Retrieval** — Maximal Marginal Relevance for diverse chunk retrieval
- **Semantic Faithfulness** — hallucination detection via embedding cosine similarity
- **HuggingFace Embeddings** — free local embeddings, no API cost
- **Streamlit Secrets** — secure API key management for deployment
