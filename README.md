# Synthetic Employee Data Generation with Llama 3.1 8B and OpenAI

This project compares **local LLM inference** using a **4-bit quantized Meta Llama 3.1 8B Instruct model** against a **hosted OpenAI model** for structured synthetic data generation.

The goal is to generate realistic synthetic employee records for a large U.S. bank, then evaluate each model on:

- **generation speed**
- **memory footprint**
- **JSON validity**
- **schema adherence**

This project is meant to demonstrate understanding of modern LLM engineering concepts such as:

- **transformer-based neural networks**
- **tokenization**
- **causal language modeling**
- **4-bit quantization**
- **GPU memory optimization**
- **sampling-based generation**
- **structured output validation**
- **local vs API-hosted inference tradeoffs**

---

## What this project does

The script:

1. Authenticates with **Hugging Face** and **OpenAI**
2. Loads **Meta Llama 3.1 8B Instruct**
3. Applies **4-bit quantization** using `BitsAndBytesConfig`
4. Loads the model with `AutoModelForCausalLM`
5. Creates a Hugging Face `text-generation` pipeline
6. Sends the same structured prompt to:
   - a **local quantized Llama model**
   - an **OpenAI hosted model**
7. Captures generation time for both
8. Validates each response against a required employee-record schema

---

## Why this project is interesting

This is not just a prompt demo.

It showcases several important LLM engineering ideas:

### 1. Transformer neural networks
The local model is a **transformer-based neural network**.  
It processes token embeddings through stacked decoder layers containing:

- **self-attention**
- **MLP / feed-forward layers**
- **layer normalization**
- **rotary positional embeddings**

From the printed model structure:

- `Embedding(128256, 4096)` → token embedding layer
- `32 x LlamaDecoderLayer` → stacked transformer decoder blocks
- `LlamaAttention` → self-attention mechanism
- `LlamaMLP` → feed-forward neural network inside each layer
- `lm_head` → output projection to vocabulary logits

This demonstrates understanding that an LLM is not magic text generation. It is a large neural network that repeatedly predicts the next token from prior context.

---

## Neural network concepts shown in this project

### Tokenization
Neural networks do not process raw text directly.  
The tokenizer converts text into integer token IDs.

Example idea:
- `"Bank analyst"` → token IDs like `[1284, 9382, 552]`

Those token IDs are then mapped into dense vectors through the model’s embedding layer.

---

### Embeddings
The embedding layer transforms token IDs into continuous vector representations.  
These vectors are what the transformer actually operates on.

In the printed model:
- `embed_tokens: Embedding(128256, 4096)`

This means:
- vocabulary size = **128256**
- embedding dimension = **4096**


- Inspired by Ed Donner's LLM Engineering course on Udemy

  
---

### Self-attention
Each transformer layer uses self-attention to determine which earlier tokens matter most when predicting the next token.

That is how the model captures relationships such as:

- employee fields staying consistent
- JSON formatting continuity
- semantic meaning across the prompt

---

### Feed-forward neural network (MLP)
After attention, each decoder block passes activations through an internal feed-forward network:

- `gate_proj`
- `up_proj`
- `down_proj`

These are dense neural network layers that help transform and refine the token representations after attention.

---

### Causal language modeling
The model is loaded with:

```python
AutoModelForCausalLM.from_pretrained(...)


