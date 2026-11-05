# GPT-2 (124M) From Scratch in PyTorch

A from-scratch implementation of the GPT-2 small architecture (124M parameters), built layer by layer starting from a dummy placeholder model. The model is **untrained**: it uses random weights, so generated text is gibberish. Training is the next step.

## What's inside

All code is in [`gpt2_model.py`](gpt2_model.py):

| Component | Description |
|---|---|
| `DummyGPTModel` | Skeleton with placeholder transformer block and layer norm |
| `LayerNorm` | Layer normalization with learnable scale and shift |
| `GELU` | Tanh approximation of GELU used by GPT-2 |
| `FeedForward` | Linear (768 → 3072), GELU, Linear (3072 → 768) |
| `ExampleDeepNeuralNetwork` | Demo showing how shortcut connections fix vanishing gradients |
| `MultiHeadAttention` | Causal multi-head self-attention, all heads in one batched pass |
| `TransformerBlock` | Pre-LayerNorm attention and feed-forward, each with a residual connection |
| `GPTModel` | Full model: embeddings, 12 transformer blocks, final norm, output head |
| `generate_text_simple` | Greedy decoding loop |

## Architecture

```
Token IDs
  -> Token embedding + Position embedding
  -> Dropout
  -> Transformer block x 12
       LayerNorm -> Masked multi-head attention -> Dropout -> + residual
       LayerNorm -> Feed-forward (GELU)         -> Dropout -> + residual
  -> Final LayerNorm
  -> Linear output head (768 -> 50,257)
  -> Logits
```

## Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Setup

Requires Python 3.9+.

```bash
python -m pip install torch tiktoken
```

## Run

```bash
python gpt2_model.py
```

The first run downloads the GPT-2 tokenizer vocabulary, so it needs internet access once.

## Expected output

- Dummy and full model output shape: `[2, 4, 50257]`
- Total parameters: 163,009,536 (124,412,160 with weight tying)
- Model size: about 621.83 MB in float32
- Shortcut test: gradients vanish toward the first layer without residuals and stay stable with them
- Generation from "Hello, I am": random tokens, since the weights are untrained

## Roadmap

- [ ] Training loop and loss function
- [ ] Sampling strategies (temperature, top-k)
- [ ] Loading pretrained GPT-2 weights
- [ ] Fine-tuning

