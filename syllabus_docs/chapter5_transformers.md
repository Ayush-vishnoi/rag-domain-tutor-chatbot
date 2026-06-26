# Chapter 5: The Transformer Architecture

## 5.1 Motivation

Before Transformers, sequence-to-sequence models relied on RNNs and LSTMs, which process tokens sequentially. This creates two problems:
1. **Slow training** — cannot parallelize across time steps
2. **Long-range dependency problem** — information from early tokens gets diluted over many steps

The Transformer (Vaswani et al., 2017 — "Attention is All You Need") solved both by relying entirely on attention mechanisms, enabling full parallelism.

---

## 5.2 Attention Mechanism

The core idea: instead of compressing the entire source sequence into a single vector, allow the decoder to "attend" to all encoder states.

### Scaled Dot-Product Attention

Given Query (Q), Key (K), Value (V) matrices:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

- **d_k** is the dimension of the key vectors (scaling prevents vanishing gradients in softmax)
- Output is a weighted sum of values, where weights are attention scores

### Multi-Head Attention

Instead of one attention function, run h parallel attention heads on linearly projected Q, K, V:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O
where head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
```

This allows the model to attend to information from different representation subspaces.

---

## 5.3 Transformer Architecture

The full Transformer has an **encoder** and a **decoder**:

### Encoder
Each encoder layer contains:
1. Multi-Head Self-Attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-Forward Network (FFN)
4. Add & Norm

### Decoder
Each decoder layer contains:
1. Masked Multi-Head Self-Attention (prevents attending to future tokens)
2. Add & Norm
3. Cross-Attention over encoder output
4. Add & Norm
5. Feed-Forward Network
6. Add & Norm

### Positional Encoding
Since attention has no notion of order, positional information is injected using sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## 5.4 BERT

BERT (Devlin et al., 2018) uses only the Transformer **encoder**. Key innovations:

- **Masked Language Modeling (MLM):** 15% of tokens are masked; model predicts them. Enables bidirectional context.
- **Next Sentence Prediction (NSP):** Model predicts if sentence B follows sentence A.

BERT is pre-trained then fine-tuned on downstream tasks (classification, NER, QA).

### BERT Variants
| Model | Parameters | Notes |
|-------|-----------|-------|
| BERT-base | 110M | 12 layers, 768 hidden |
| BERT-large | 340M | 24 layers, 1024 hidden |
| RoBERTa | 125M | Removes NSP, trains longer |
| DistilBERT | 66M | 40% smaller, 97% of BERT performance |

---

## 5.5 GPT Family

GPT models use only the Transformer **decoder** with causal (left-to-right) masking. They are trained with a simple next-token prediction objective.

- **GPT-1** (2018): 117M parameters, introduced unsupervised pre-training + supervised fine-tuning
- **GPT-2** (2019): 1.5B parameters, zero-shot task transfer
- **GPT-3** (2020): 175B parameters, few-shot in-context learning
- **GPT-4** (2023): Multimodal, state-of-the-art on most benchmarks

---

## 5.6 Key Differences: BERT vs GPT

| Property | BERT | GPT |
|----------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Training objective | MLM + NSP | Next token prediction |
| Context | Bidirectional | Unidirectional (left-to-right) |
| Best for | Classification, NER, QA | Text generation |

---

## 5.7 Summary

- Transformers replace recurrence with attention, enabling parallelism and better long-range dependencies
- Multi-head attention lets the model jointly attend to different positions
- BERT uses bidirectional encoders for understanding tasks; GPT uses decoders for generation
- Most modern NLP systems are built on Transformer variants
