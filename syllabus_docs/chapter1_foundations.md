# Chapter 1: Foundations of NLP

## 1.1 Text Preprocessing

Raw text must be cleaned and normalized before any NLP task.

### Tokenization
Splitting text into individual units (tokens) — words, subwords, or characters.
- **Word tokenization:** "I love NLP" → ["I", "love", "NLP"]
- **Subword tokenization (BPE):** Used in modern LLMs to handle unknown words
- **Sentence tokenization:** Splitting a document into sentences

### Stemming vs Lemmatization
- **Stemming:** Crudely chops word endings. "running" → "run", "studies" → "studi" (may not be a real word)
- **Lemmatization:** Returns the dictionary base form using vocabulary and morphological analysis. "studies" → "study"

### Stopword Removal
Removing high-frequency, low-information words (e.g., "the", "is", "at") to reduce noise.

### Normalization
- Lowercasing
- Removing punctuation and special characters
- Expanding contractions ("don't" → "do not")

---

## 1.2 Bag-of-Words (BoW)

Represents a document as a vector of word counts, ignoring order.

Example:
- Doc 1: "I love NLP" → {I:1, love:1, NLP:1}
- Doc 2: "I love ML" → {I:1, love:1, ML:1}

Limitation: No word order, no semantic meaning.

---

## 1.3 TF-IDF

**Term Frequency-Inverse Document Frequency** — weights words by how important they are to a document relative to the corpus.

```
TF(t, d)  = count of term t in document d / total terms in d
IDF(t)    = log(N / df(t))   where N = total docs, df(t) = docs containing t
TF-IDF    = TF * IDF
```

- Common words across all docs (e.g., "the") get low IDF → low score
- Rare but relevant words get high IDF → high score

Used in: document retrieval, keyword extraction, search engines.

---

## 1.4 N-gram Language Models

An **N-gram** is a contiguous sequence of N tokens.
- Unigram (N=1): "cat"
- Bigram (N=2): "black cat"
- Trigram (N=3): "the black cat"

### Language Model
Assigns probability to a sequence of words using the chain rule + Markov assumption:

```
P(w1, w2, ..., wn) ≈ ∏ P(wi | wi-N+1, ..., wi-1)
```

Bigram model: P(cat | black) = count("black cat") / count("black")

### Limitations
- Sparsity: many N-grams never seen in training
- Fixed context window

---

## 1.5 Evaluation Metrics

### Perplexity
Measures how well a language model predicts a test set. Lower = better.
```
Perplexity = 2^(-1/N * sum(log2 P(wi)))
```

### BLEU (Bilingual Evaluation Understudy)
Used for machine translation. Measures n-gram overlap between generated and reference text.
- Ranges from 0 to 1 (higher is better)
- Penalizes overly short outputs (brevity penalty)

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Used for summarization. Measures recall of n-gram overlap.
- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest common subsequence

---

## 1.6 Summary
- Text preprocessing (tokenization, stemming, lemmatization) is the first step in any NLP pipeline
- BoW and TF-IDF are simple but effective bag-of-words representations
- N-gram language models estimate word sequence probabilities using the Markov assumption
- Perplexity, BLEU, and ROUGE are standard NLP evaluation metrics
