# Model Architecture

## Overview

The model is a three-stage pipeline that operates over a sliding window of consecutive battery discharge cycles:

1. **Cycle encoder** — maps each variable-length voltage/current signal to a fixed-size latent vector.
2. **Aggregator** — cross-cycle transformer attention followed by attention pooling, collapsing the sequence of per-cycle latents into a single context vector.
3. **Forecast head** — takes the context vector and sinusoidal offset embeddings to predict future discharge capacities at arbitrary cycle offsets.

The model operates entirely on a sliding window of consecutive cycles provided by the dataset. It never sees the full battery lifecycle during a single forward pass.

---

## Stage 1: Cycle Encoder (`CycleEncoder`)

**Goal:** compress a variable-length discharge signal `(T, 2)` → fixed latent vector `(latent_dim,)`.

### Sub-components

**1. Conv1D feature extractor (`ConvFeatureExtractor`)**

A stack of 1D convolutions operating along the time axis, with:
- `GroupNorm` after each convolution (with GroupNorm group count clamped so it always divides the channel count).
- `GELU` activations.
- `same` padding per layer (`padding = kernel_size // 2`), so spatial output length is determined only by stride.
- Input is always 2 channels (voltage, current). This is not configurable.

The convolution stack progressively extracts local features and can downsample the time axis via strides (default: first layer stride 1, second stride 2).

**Nuance — mask propagation:** The validity mask is propagated through each strided convolution using a sliding-window OR: a downsampled position is valid if *any* input position in its receptive field was valid. This is implemented as a 1D convolution of the float mask with a ones kernel. This ensures padded NaN positions are excluded from attention but the mask boundary reflects actual signal coverage.

**2. Sinusoidal positional encoding**

Applied to the downsampled hidden states after the conv stack. Uses the standard sinusoidal schedule. The positional encoding cache is built on the fly and extended as needed — there is no fixed maximum.

**3. Transformer blocks (`SignalTransformerBlock`)**

One or more standard pre-norm transformer blocks with:
- `LayerNorm` → multi-head self-attention → dropout → residual.
- `LayerNorm` → feed-forward (Linear → GELU → Dropout → Linear) → dropout → residual.
- Padding positions are zeroed out after each attention and feed-forward operation to prevent mask leakage.
- `key_padding_mask` is passed to `nn.MultiheadAttention` to suppress padding positions in attention scores.

**4. Masked attention pooling (`MaskedAttentionPooling`)**

Collapses the variable-length sequence of hidden states into a fixed-size vector using learned attention weights, one set of weights per pooling head. The pooling scores are masked so padded positions receive `-inf` before softmax. The `num_heads` pooled vectors are concatenated, giving output size `d_model * pooling_heads`.

**5. Projection head**

A single `Linear` followed by `LayerNorm` projects the concatenated pooled output down to the `latent_dim`. The LayerNorm stabilizes the latent space, especially important given the pooling multi-head concatenation.

### Encoder output

Each input cycle produces one latent vector of shape `(latent_dim,)`. For a batch of windows with `W` cycles each, the encoder is applied flat across all `batch * W` cycles simultaneously for efficiency.

---

## Stage 2: Aggregator

**Goal:** collapse the sequence of per-cycle latents `(B, W, latent_dim)` into a single context vector `(B, out_dim)`.

The aggregator lives inside `CapacityForecastModel` and consists of:

### 1. Cross-cycle transformer with RoPE

RoPE is applied first to the cycle-latent sequence with `RotaryPositionalEncoding(latent_dim, base=rotary_base)`, then one or more standard `SignalTransformerBlock` layers operate over the `W` cycle positions with non-causal self-attention.

In other words, positional information is injected directly into `(B, W, latent_dim)` before attention, and the transformer stack remains the same pre-norm block used in the encoder (`nn.MultiheadAttention` + FFN + residuals).

The `rotary_base` parameter (default `10000.0`) controls the RoPE frequency spread. RoPE caches cosine/sine tables and grows them on demand when sequence length `W` exceeds the current cache.

### 2. Attention pooling

`MaskedAttentionPooling(latent_dim, pooling_heads)` learns `pooling_heads` separate attention-weighted averages over the `W` positions, producing a vector of size `latent_dim * pooling_heads`.

### 3. Projection

`Linear(latent_dim * pooling_heads → out_dim) + LayerNorm` projects the concatenated pooled output to `out_dim`. This is typically larger than the encoder's `latent_dim` to give the forecast head a richer representation to condition on.

### Why a separate aggregator output dim?

The aggregator output feeds the forecast head which must predict capacity at arbitrary future offsets. A larger output dim gives the context vector more dimensions to work with, which benefits generalization — particularly when predicting far into the future beyond the training `pred_seq_len`.

---

## Stage 3: Forecast Head

**Goal:** predict scalar capacity at arbitrary future cycle offsets given the context vector.

### Design

1. **Offset embedding** — `SinusoidalEmbedding(offset_embedding_dim)` maps integer cycle offsets `[0, 1, ..., n-1]` to dense vectors on the fly. The `offset_embedding_dim` is configured independently from the aggregator's `out_dim`.

2. **Concatenation** — the context vector is broadcast across the `n` offsets and concatenated with offset embeddings: `(B, n, out_dim + offset_embedding_dim)`.

3. **MLP** — `Linear(out_dim + offset_embedding_dim → hidden_dim) → GELU → Linear(hidden_dim → 1)` produces one scalar capacity prediction per offset.

### Arbitrary horizon

The head can predict at any number of offsets at inference time, not limited to the training `pred_seq_len`. The sinusoidal offset embedding generalizes to unseen positions.

---

## Masks

Two masks are maintained throughout the pipeline:

| Mask | Shape | Meaning |
|---|---|---|
| `signal_mask` | `(B, W, T)` | valid sample positions within each cycle |
| `sequence_mask` | `(B, W)` | valid cycle positions within each window |

All masking is done via zeroing after attention/FFN operations and `key_padding_mask` in attention layers.
