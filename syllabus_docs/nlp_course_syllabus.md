# NLP Course Syllabus - CS 5340

## Course Overview
This course covers Natural Language Processing (NLP) theory and practice. Students will learn core NLP techniques, modern deep learning approaches, and how to build real-world NLP systems.

**Instructor:** Dr. Jane Smith  
**Credits:** 3  
**Prerequisites:** Linear Algebra, Python Programming, Basic ML

---

## Module 1: Foundations of NLP
- Text preprocessing: tokenization, stemming, lemmatization
- Regular expressions and pattern matching
- Bag-of-Words (BoW) and TF-IDF representations
- N-gram language models
- Evaluation metrics: perplexity, BLEU, ROUGE

## Module 2: Classical NLP
- Part-of-Speech (POS) tagging using Hidden Markov Models (HMM)
- Named Entity Recognition (NER)
- Dependency parsing and constituency parsing
- Sentiment analysis with Naive Bayes and Logistic Regression
- Sequence labeling with Conditional Random Fields (CRF)

## Module 3: Word Embeddings and Representations
- Word2Vec: Skip-gram and CBOW architectures
- GloVe (Global Vectors for Word Representation)
- FastText for subword representations
- Evaluation: word similarity and analogy tasks
- Contextual vs. static embeddings

## Module 4: Sequence Models
- Recurrent Neural Networks (RNN) and vanishing gradients
- Long Short-Term Memory (LSTM) networks
- Gated Recurrent Units (GRU)
- Bidirectional RNNs
- Sequence-to-sequence (Seq2Seq) models with attention

## Module 5: Transformers and Attention
- The Attention Mechanism (Bahdanau, Luong)
- Transformer architecture: encoder-decoder, multi-head attention, positional encoding
- BERT: Bidirectional Encoder Representations from Transformers
- GPT family: autoregressive language modeling
- Fine-tuning pre-trained models

## Module 6: Large Language Models (LLMs)
- Scaling laws and emergent abilities
- Instruction tuning and RLHF (Reinforcement Learning from Human Feedback)
- Prompt engineering: zero-shot, few-shot, chain-of-thought
- LLM evaluation: benchmarks (MMLU, HellaSwag, TruthfulQA)
- Hallucination in LLMs: causes and mitigation

## Module 7: Retrieval-Augmented Generation (RAG)
- Motivation: knowledge cutoff and hallucination in LLMs
- Dense retrieval: bi-encoders and cross-encoders
- Vector databases: FAISS, Pinecone, Chroma
- RAG pipeline: document loading, chunking, embedding, retrieval, generation
- Advanced RAG: HyDE, re-ranking, query expansion
- Evaluation: faithfulness, relevancy, context precision/recall

## Module 8: NLP Applications
- Machine Translation
- Question Answering systems
- Text Summarization (extractive vs. abstractive)
- Dialogue systems and chatbots
- Information Extraction

---

## Assignments
- Assignment 1: Build a TF-IDF based document retrieval system (Week 3)
- Assignment 2: Train Word2Vec on a custom corpus (Week 5)
- Assignment 3: Fine-tune BERT for text classification (Week 8)
- Assignment 4: Build a RAG chatbot over a domain corpus (Week 11) — **Final Project**

## Grading
- Assignments: 40%
- Midterm Exam: 20%
- Final Project: 30%
- Participation: 10%
