# Model Architecture

## Overview

The model is a three-stage pipeline that operates over a sequence of battery discharge cycles:

1. **Cycle encoder** — maps each variable-length voltage/current signal to a fixed-size latent vector.
2. **Latent predictor** — autoregressively models degradation dynamics across the latent sequence and predicts the next-cycle latent residual.
3. **Capacity decoder** — maps the predicted next-cycle latent to a capacity estimate (mean + log-variance).

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

Applied to the downsampled hidden states after the conv stack. Uses the standard sinusoidal schedule with a configurable `max_signal_positions` cap. This cap must be at least as large as the longest downsampled signal length in the dataset (use the notebook analysis cell to estimate this).

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

## Stage 2: Latent Predictor (`LatentPredictor`)

**Goal:** given the sequence of encoded latents `(W, latent_dim)`, predict the next-cycle latent for each position autoregressively.

### Design

A causal transformer (pre-norm, like the signal encoder) with:
- `CausalSelfAttention` using Rotary Positional Embeddings (RoPE) and an upper-triangular causal mask.
- Key padding mask applied to suppress padding positions (shorter windows in a batch pad to the max window length).
- `FeedForward` block with `GELU` activations.

### Rotary positional embeddings (RoPE)

RoPE is applied per-head to queries and keys. This avoids adding absolute positions to the latent values and gives the predictor relative position awareness which generalises to unseen window offsets. `rotary_base = 10000.0` controls the frequency spread; increasing it extends positional resolution at the cost of shorter effective range.

### Residual prediction

The predictor does **not** directly predict the next latent. It predicts a **residual**:

```
predicted_next_latent[t] = latent[t] + residual_head(hidden[t])
```

This means the model learns to predict the *change* in latent state from cycle to cycle, not the absolute next state. This is a strong inductive bias for battery degradation: cycles change slowly, so the residual is small and well-conditioned early in training.

**Nuance — time offset:** The predictor hidden states at position `t` are used to predict the latent at `t+1`. This means the output is `(W-1, latent_dim)` aligning with `latents[:, 1:]` on the target side. The last latent position never produces a prediction (there is nothing to supervise it against in a window).

---

## Stage 3: Capacity Decoder (`CapacityDecoder`)

A two-layer MLP:
- `Linear(latent_dim → hidden_dim)` → `GELU` → `Linear(hidden_dim → 2)`

The two output values are:
- `capacity_mean`: predicted normalized capacity for the next cycle.
- `capacity_logvar`: log-variance of the (optional) Gaussian predictive distribution.

The decoder is applied position-wise to `predicted_next_latent`, so it produces one (mean, logvar) pair per predicted cycle position.

**Nuance — logvar when Gaussian is disabled:** When `loss.learn_gaussian_likelihood = false`, `capacity_logvar` is produced by the model but is not included in the loss. The decoder head still outputs it; it simply does not receive gradient from the capacity objective. This is intentional: it keeps the model structure identical between modes so checkpoints are compatible.

---

## Masks

Three masks are maintained throughout the pipeline:

| Mask | Shape | Meaning |
|---|---|---|
| `signal_mask` | `(B, W, T)` | valid sample positions within each cycle |
| `sequence_mask` | `(B, W)` | valid cycle positions within each window (some windows may be shorter due to batching) |
| `prediction_mask` | `(B, W-1)` | positions where a next-cycle prediction exists (both current and next cycle are valid) |
| `target_capacity_mask` | `(B, W-1)` | positions where capacity supervision is available (prediction mask AND discharge capacity was valid) |

All losses are computed only over positions where the respective mask is true.

---

## Why latent residual prediction?

Predicting the residual instead of the full next latent has several advantages:
1. **Gradient stability early in training:** residuals start near zero, so the predictor loss starts low and gradients are small and consistent.
2. **Natural inductive bias:** battery degradation is approximately monotonic and slow; the latent trajectory has low curvature, so residuals are small.
3. **Easier to learn than absolute prediction:** the model only needs to learn the direction and magnitude of change, not reconstruct the full representation from scratch.