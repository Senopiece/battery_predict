# Model Notes

The implementation follows a two-stage design:

1. A cycle encoder maps variable-length voltage/current traces to a fixed latent vector.
2. An autoregressive transformer models degradation across latent sequences and predicts the next latent residual.

The decoder outputs two values per predicted cycle:

- capacity mean
- capacity log-variance

Training uses:

- latent MSE on predicted next-step latent vectors
- Gaussian negative log-likelihood on normalized capacity targets

Teacher forcing is the active training mode in v1. Scheduled-sampling and rollout hyperparameters remain configurable but inactive.