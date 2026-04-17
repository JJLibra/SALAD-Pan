# Utils

Helper scripts and metric functions used by training and evaluation.

## Files

- `get_training_vae_data.py`
  - Builds Stage-I VAE training data from source datasets.
  - Expected output: single-channel H5 used by `train_vae.py`.
- `calibrate_vae_scaling_factor.py`
  - Utility for calibrating VAE latent scaling factor.
- `metrics.py`
  - Remote-sensing quality metrics used in validation/inference:
    `Q4`, `Q8`, `SAM`, `ERGAS`, `SCC`.

## Typical Workflow

1. Prepare H5 training data with `get_training_vae_data.py`.
2. Train Stage-I VAE (`train_vae.py`).
3. Train Stage-II diffusion adapter (`train_diffusion.py`).
4. Evaluate with `inference.py` (which imports metrics from this folder).

## Note

Metric implementations here are treated as project ground truth during validation/inference.  
If you replace them, keep formulas and input ranges consistent to avoid incomparable results.
