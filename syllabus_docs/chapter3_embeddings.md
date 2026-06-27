# Chapter 3: Word Embeddings and Representations

## 3.1 Why Embeddings?

BoW and TF-IDF treat words as independent symbols — "king" and "queen" are as unrelated as "king" and "banana". Word embeddings solve this by mapping words to dense vectors in a continuous space where **semantic similarity = geometric proximity**.

---

## 3.2 Word2Vec

Word2Vec (Mikolov et al., 2013) learns word embeddings by training a shallow neural network to predict context words.

### Two Architectures

**Skip-gram:** Given a center word, predict surrounding context words.
- Input: "NLP" → Predict: ["love", "study", "deep"]
- Better for rare words

**CBOW (Continuous Bag of Words):** Given context words, predict the center word.
- Input: ["love", "study", "deep"] → Predict: "NLP"
- Faster to train

### Training Objective
Maximize the probability of context words given the center word (Skip-gram):
```
maximize ∑ ∑ log P(context_word | center_word)
```

### Key Properties
Word2Vec embeddings capture **analogical relationships**:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

---

## 3.3 GloVe (Global Vectors)

GloVe (Pennington et al., 2014) learns embeddings from **global word co-occurrence statistics** across the entire corpus.

- Builds a co-occurrence matrix X where X_ij = how often word j appears in context of word i
- Learns vectors such that: **w_i · w_j ≈ log(X_ij)**

Combines the benefits of:
- Matrix factorization methods (captures global statistics)
- Word2Vec (captures local context)

GloVe often outperforms Word2Vec on word similarity benchmarks.

---

## 3.4 FastText

FastText (Bojanowski et al., 2017) extends Word2Vec by representing words as bags of **character n-grams**.

- "where" → {wh, whe, her, ere, re, <where>}
- The word vector = sum of its subword vectors

**Advantages:**
- Handles **out-of-vocabulary (OOV)** words — can embed words never seen in training
- Better for morphologically rich languages
- Better representations for rare words

---

## 3.5 Evaluation of Word Embeddings

### Word Similarity
Compare cosine similarity of word pairs against human judgments.
- Datasets: WordSim-353, SimLex-999

### Word Analogy
Test whether: vector(A) - vector(B) + vector(C) ≈ vector(D)
- Example: "Paris" - "France" + "Germany" ≈ "Berlin"

---

## 3.6 Static vs Contextual Embeddings

| Property | Static (Word2Vec, GloVe) | Contextual (BERT, GPT) |
|----------|--------------------------|------------------------|
| One vector per word | ✅ Yes | ❌ No — depends on context |
| "bank" (river vs. finance) | Same vector | Different vectors |
| Captures polysemy | ❌ No | ✅ Yes |
| Computational cost | Low | High |

Static embeddings assign the same vector to a word regardless of context. Contextual embeddings (from Transformers) produce different representations depending on the surrounding sentence.

---

## 3.7 Summary
- Word embeddings map words to dense vectors capturing semantic relationships
- Word2Vec uses local context (Skip-gram / CBOW) to learn embeddings
- GloVe uses global co-occurrence statistics
- FastText uses subword n-grams, handling OOV words effectively
- Contextual embeddings (BERT) outperform static embeddings for most downstream tasks
