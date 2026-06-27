# Chapter 4: Sequence Models

## 4.1 Recurrent Neural Networks (RNN)

An RNN processes sequential data by maintaining a **hidden state** that carries information from previous time steps.

```
h_t = tanh(W_h * h_(t-1) + W_x * x_t + b)
y_t = W_y * h_t
```

- h_t: hidden state at time t (memory of past inputs)
- x_t: input at time t
- Same weights W shared across all time steps (parameter efficiency)

### Vanishing Gradient Problem
During backpropagation through time (BPTT), gradients are multiplied repeatedly. If weights < 1, gradients shrink exponentially → **early time steps receive near-zero gradients** → model fails to learn long-range dependencies.

---

## 4.2 Long Short-Term Memory (LSTM)

LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient problem with a **cell state** (long-term memory) and three **gates**:

### Gates
- **Forget gate:** Decides what to erase from cell state
  ```f_t = σ(W_f * [h_(t-1), x_t] + b_f)```
- **Input gate:** Decides what new info to write to cell state
  ```i_t = σ(W_i * [h_(t-1), x_t] + b_i)```
- **Output gate:** Decides what to output from cell state
  ```o_t = σ(W_o * [h_(t-1), x_t] + b_o)```

### Cell State Update
```
C_t = f_t * C_(t-1) + i_t * tanh(W_C * [h_(t-1), x_t])
h_t = o_t * tanh(C_t)
```

The cell state acts as a "conveyor belt" — gradients flow through it with minimal decay, enabling learning of long-range dependencies.

---

## 4.3 Gated Recurrent Unit (GRU)

GRU (Cho et al., 2014) simplifies LSTM with only **two gates**, merging cell state and hidden state:

- **Reset gate:** Controls how much past state to forget
- **Update gate:** Controls how much past state to keep

GRU has fewer parameters than LSTM and trains faster, with comparable performance on most tasks.

---

## 4.4 Bidirectional RNNs

A standard RNN only sees past context. A **Bidirectional RNN** runs two RNNs:
- Forward RNN: left → right
- Backward RNN: right → left

Final hidden state = concatenation of both directions, capturing both past and future context.

Widely used in NER, POS tagging, and other sequence labeling tasks where full context is available.

---

## 4.5 Sequence-to-Sequence (Seq2Seq) Models

Seq2Seq (Sutskever et al., 2014) uses an **encoder-decoder** architecture for tasks where input and output are both sequences (e.g., machine translation, summarization).

### Architecture
- **Encoder:** Reads input sequence, compresses into a context vector (final hidden state)
- **Decoder:** Generates output sequence token by token, conditioned on context vector

### Limitation
The entire input is compressed into a single fixed-size vector — information bottleneck for long sequences.

### Attention Mechanism (Bahdanau, 2015)
Instead of a single context vector, the decoder **attends** to all encoder hidden states at each decoding step:

```
e_ij  = score(s_(i-1), h_j)          # alignment score
α_ij  = softmax(e_ij)                 # attention weights
c_i   = ∑ α_ij * h_j                 # context vector for step i
```

Attention allows the model to focus on relevant parts of the input at each output step. This was the precursor to the Transformer's self-attention.

---

## 4.6 Comparison

| Model | Long-range deps | Parallelizable | Parameters |
|-------|----------------|----------------|------------|
| RNN | ❌ Poor | ❌ No | Low |
| LSTM | ✅ Good | ❌ No | High |
| GRU | ✅ Good | ❌ No | Medium |
| Transformer | ✅ Excellent | ✅ Yes | Very High |

---

## 4.7 Summary
- RNNs process sequences with shared weights but suffer from vanishing gradients
- LSTMs use gating mechanisms and a cell state to learn long-range dependencies
- GRUs simplify LSTMs with two gates and are computationally cheaper
- Bidirectional RNNs capture both past and future context
- Seq2Seq with attention was the dominant architecture before Transformers
