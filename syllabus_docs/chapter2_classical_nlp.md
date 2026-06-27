# Chapter 2: Classical NLP

## 2.1 Part-of-Speech (POS) Tagging

Assigns grammatical labels (noun, verb, adjective, etc.) to each token.

Example: "The cat sat" → [DT, NN, VBD]

### Hidden Markov Model (HMM) for POS Tagging
HMM models POS tagging as a sequence labeling problem with two components:
- **Emission probability:** P(word | tag) — how likely is a word given a tag
- **Transition probability:** P(tag_i | tag_i-1) — how likely is one tag to follow another

**Viterbi algorithm** finds the most probable tag sequence efficiently using dynamic programming.

```
P(tags | words) ∝ ∏ P(word_i | tag_i) * P(tag_i | tag_i-1)
```

---

## 2.2 Named Entity Recognition (NER)

Identifies and classifies named entities in text into categories:
- **PER** (Person): "Elon Musk"
- **ORG** (Organization): "OpenAI"
- **LOC** (Location): "San Francisco"
- **DATE/TIME:** "Monday", "2024"

### BIO Tagging Scheme
- **B-** (Beginning of entity): first token of entity
- **I-** (Inside entity): continuation token
- **O** (Outside): not an entity

Example: "Barack Obama visited Paris"
→ B-PER, I-PER, O, B-LOC

---

## 2.3 Parsing

### Dependency Parsing
Identifies grammatical relationships between words (subject, object, modifier).
- "The cat chased the mouse" → cat←nsubj←chased→dobj→mouse

### Constituency Parsing
Breaks sentence into nested phrases (NP, VP, PP):
```
(S (NP The cat) (VP chased (NP the mouse)))
```

---

## 2.4 Sentiment Analysis

Classifies text as positive, negative, or neutral.

### Naive Bayes Classifier
Uses Bayes' theorem with bag-of-words features:
```
P(class | text) ∝ P(class) * ∏ P(word_i | class)
```
- Fast, simple, works well for text classification
- Assumes word independence (naive assumption)

### Logistic Regression
Learns a linear decision boundary over TF-IDF features. More powerful than Naive Bayes, handles feature correlations better.

---

## 2.5 Conditional Random Fields (CRF)

CRF is a discriminative sequence labeling model — better than HMM for NER and POS tagging because it:
- Models P(tags | words) directly (discriminative, not generative)
- Can use arbitrary overlapping features (previous word, suffix, capitalization, etc.)
- Does not assume conditional independence of observations

```
P(y | x) = (1/Z) * exp(∑ λ_k * f_k(y_i, y_i-1, x, i))
```

CRFs were the state-of-the-art for sequence labeling before deep learning.

---

## 2.6 Summary
- HMMs model sequence labeling generatively using emission and transition probabilities
- NER uses BIO tagging to label entity spans in text
- Dependency and constituency parsing reveal grammatical structure
- Naive Bayes and Logistic Regression are effective classical classifiers for sentiment analysis
- CRFs are discriminative sequence models that outperform HMMs for structured prediction tasks
