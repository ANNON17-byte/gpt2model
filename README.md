# GPT-2 (124M) From Scratch in PyTorch

<<<<<<< HEAD
A from-scratch implementation of the GPT-2 small architecture (124M parameters), built layer by layer. The model is **untrained**: it uses random weights, so generated text is gibberish. Training is the next step.
=======
A from-scratch implementation of the GPT-2 small architecture (124M parameters) in PyTorch. The script downloads OpenAI's pretrained GPT-2 weights, loads them into the model, and generates text.
>>>>>>> 51d1df3 (Update GPT-2 model implementation)

## What's inside

All code is in [`gpt2_model.py`](gpt2_model.py):

| Component | Description |
|---|---|
| `LayerNorm` | Layer normalization with learnable scale and shift |
| `GELU` | Tanh approximation of GELU used by GPT-2 |
<<<<<<< HEAD
| `FeedForward` | Linear (768 → 3072), GELU, Linear (3072 → 768) |
=======
| `FeedForward` | Linear (768 -> 3072), GELU, Linear (3072 -> 768) |
>>>>>>> 51d1df3 (Update GPT-2 model implementation)
| `MultiHeadAttention` | Causal multi-head self-attention, all heads in one batched pass |
| `TransformerBlock` | Pre-LayerNorm attention and feed-forward, each with a residual connection |
| `GPTModel` | Full model: embeddings, 12 transformer blocks, final norm, output head |
| `generate_text_simple` | Greedy decoding loop |
| `download_and_load_gpt2` | Downloads the checkpoint files and returns the settings and parameters |
| `download_file` | Downloads a single file with a progress bar, skipping files already downloaded |
| `load_gpt2_params_from_tf_ckpt` | Reads the TensorFlow checkpoint into a nested dictionary of NumPy arrays |
| `assign` | Checks that shapes match, then wraps the pretrained array as a PyTorch parameter |
| `load_weights_into_gpt` | Copies the pretrained weights into `GPTModel` |

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

The pretrained OpenAI weights include bias terms in the query, key and value layers, so the script sets `qkv_bias` to `True` before loading them.

## Setup

Requires Python 3.9+.

```bash
python -m pip install torch tiktoken tensorflow requests tqdm numpy
```

## Run

```bash
python gpt2_model.py
```

On the first run the script downloads the GPT-2 tokenizer vocabulary and the 124M model files (about 500 MB) into `gpt2/124M`. Later runs reuse the downloaded files.

## What it does

1. Encodes the prompt "Hello, I am" with the GPT-2 tokenizer.
2. Downloads and reads the pretrained 124M checkpoint.
3. Builds `GPTModel` with `qkv_bias=True` and loads the pretrained weights into it.
4. Generates 25 new tokens with greedy decoding and prints the text.

Because the weights are pretrained, the output is coherent English. Greedy decoding always picks the most likely token, so the same prompt gives the same output every time.
