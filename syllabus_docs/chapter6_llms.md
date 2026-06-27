# Chapter 6: Large Language Models (LLMs)

## 6.1 Scaling Laws

Kaplan et al. (2020) showed that LLM performance follows predictable **power laws** with respect to:
- **Model size** (number of parameters)
- **Dataset size** (number of training tokens)
- **Compute budget** (FLOPs)

Key finding: Larger models are more sample-efficient — they need fewer examples to reach the same loss. This justified the trend of scaling models from millions to hundreds of billions of parameters.

**Chinchilla scaling law (Hoffmann et al., 2022):** For a given compute budget, the optimal strategy is to scale model size and data size equally. GPT-3 was undertrained relative to its size.

---

## 6.2 Emergent Abilities

Emergent abilities are capabilities that appear suddenly at a certain scale and are not present in smaller models:
- Multi-step arithmetic
- Chain-of-thought reasoning
- In-context learning (few-shot)
- Code generation

These abilities are not explicitly trained — they emerge from scale.

---

## 6.3 Instruction Tuning

Pre-trained LLMs predict the next token but don't follow instructions well. **Instruction tuning** fine-tunes the model on (instruction, response) pairs to make it helpful:

```
Input:  "Summarize the following article: ..."
Output: "The article discusses..."
```

Models like FLAN-T5, InstructGPT, and Llama-2-chat are instruction-tuned. This dramatically improves zero-shot task performance.

---

## 6.4 RLHF (Reinforcement Learning from Human Feedback)

RLHF (Ouyang et al., 2022 — InstructGPT) aligns LLMs with human preferences in 3 steps:

**Step 1: Supervised Fine-Tuning (SFT)**
Fine-tune on high-quality human-written demonstrations.

**Step 2: Reward Model Training**
Collect human preference data (rank model outputs A vs B). Train a reward model to predict human preference score.

**Step 3: PPO (Proximal Policy Optimization)**
Fine-tune the LLM using RL — maximize reward model score while staying close to the SFT model (KL divergence penalty).

RLHF makes models more helpful, harmless, and honest (HHH).

---

## 6.5 Prompt Engineering

Since LLMs are sensitive to input phrasing, prompt design is critical.

### Zero-shot Prompting
Ask the model directly without examples:
```
Classify the sentiment: "I loved this movie!"
```

### Few-shot Prompting
Provide a few input-output examples in the prompt:
```
"I loved it!" → Positive
"It was terrible." → Negative
"I loved this movie!" → ?
```

### Chain-of-Thought (CoT) Prompting
Ask the model to reason step by step:
```
Q: Roger has 5 balls. He buys 2 more cans of 3 balls each. How many does he have?
A: Roger started with 5. Bought 2*3=6 more. 5+6=11. Answer: 11.
```
CoT dramatically improves performance on arithmetic and multi-step reasoning tasks.

### System Prompts
Instructions given to the model before the conversation to set behavior, persona, or constraints.

---

## 6.6 LLM Evaluation Benchmarks

| Benchmark | What it tests |
|-----------|--------------|
| MMLU | Massive Multitask Language Understanding — 57 subjects |
| HellaSwag | Commonsense NLI / sentence completion |
| TruthfulQA | Tendency to generate false but believable answers |
| HumanEval | Code generation (pass@k) |
| GSM8K | Grade school math word problems |
| BIG-Bench | 200+ diverse tasks beyond standard benchmarks |

---

## 6.7 Hallucination in LLMs

LLMs generate fluent but factually incorrect text — called **hallucination**.

### Causes
- Training data contains noise and contradictions
- Model learns statistical patterns, not facts
- Over-reliance on parametric memory
- No grounding mechanism during generation

### Types
- **Factual hallucination:** Incorrect facts ("Einstein won the Nobel Prize in 1925" — actual year: 1921)
- **Faithfulness hallucination:** Generated text contradicts the provided context
- **Fabrication:** Entirely made-up entities, citations, or events

### Mitigation Strategies
- **RAG (Retrieval-Augmented Generation):** Ground responses in retrieved documents
- **RLHF:** Train model to prefer honest, calibrated responses
- **Self-consistency:** Sample multiple outputs and select the most consistent
- **Verification chains:** Ask model to verify its own claims

---

## 6.8 Summary
- LLM performance scales predictably with model size, data, and compute (scaling laws)
- Emergent abilities appear at scale and are not explicitly trained
- Instruction tuning and RLHF align pre-trained LLMs to be helpful and safe
- Prompt engineering (zero-shot, few-shot, CoT) significantly affects LLM performance
- Hallucination is a core limitation of LLMs — RAG is the primary mitigation strategy
