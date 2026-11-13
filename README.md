# GPT-2 (124M) From Scratch in PyTorch

A from-scratch implementation of the GPT-2 small architecture (124M parameters) in PyTorch. The implementation builds the model layer by layer and downloads OpenAI's pretrained GPT-2 weights to generate coherent text.

## What's Inside

All code is in [`gpt2_model.py`](https://github.com/ANNON17-byte/gpt2model/blob/main/gpt2_model.py).

| Component | Description |
|---|---|
| `LayerNorm` | Layer normalization with learnable scale and shift |
| `GELU` | Tanh approximation of GELU used by GPT-2 |
| `FeedForward` | Linear (768 → 3072), GELU, Linear (3072 → 768) |
| `MultiHeadAttention` | Causal multi-head self-attention |
| `TransformerBlock` | Pre-LayerNorm attention and feed-forward with residual connections |
| `GPTModel` | Full model with embeddings, 12 transformer blocks, final normalization and output head |
| `generate_text_simple` | Greedy decoding loop |
| `download_and_load_gpt2` | Downloads the pretrained GPT-2 checkpoint |
| `load_weights_into_gpt` | Loads pretrained weights into the model |

## Architecture

```text
Token IDs
   ↓
Token Embedding + Position Embedding
   ↓
Dropout
   ↓
Transformer Block × 12
   ├── LayerNorm
   ├── Masked Multi-Head Attention
   ├── Residual Connection
   ├── LayerNorm
   ├── Feed-Forward Network (GELU)
   └── Residual Connection
   ↓
Final LayerNorm
   ↓
Linear Output Head (768 → 50,257)
   ↓
Logits
```

## Configuration

GPT-2 124M configuration:

| Parameter | Value |
|---|---:|
| Vocabulary Size | 50,257 |
| Context Length | 1,024 |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Dropout | 0.1 |
| QKV Bias | False (True when loading pretrained weights) |

## Setup

Requires Python 3.9+.

Install dependencies:

```bash
python -m pip install torch tiktoken tensorflow requests tqdm numpy
```

## Run

```bash
python gpt2_model.py
```

On the first run, the script downloads the GPT-2 tokenizer vocabulary and pretrained 124M checkpoint files (approximately 500 MB) into `gpt2/124M`.

Subsequent runs reuse the downloaded files.

## What It Does

1. Encodes the prompt `"Hello, I am"` using the GPT-2 tokenizer.
2. Downloads and reads the pretrained 124M checkpoint.
3. Builds the GPT-2 model with `qkv_bias=True`.
4. Loads OpenAI's pretrained weights into the model.
5. Generates 25 new tokens using greedy decoding.
6. Decodes and prints the generated text.

Since the model uses pretrained weights, it generates coherent English text. Greedy decoding selects the most probable next token at every step, producing deterministic output for the same prompt.

## Future Improvements

- Implement model training from scratch.
- Experiment with temperature-based sampling.
- Add top-k and top-p sampling.
- Fine-tune GPT-2 on custom datasets.
