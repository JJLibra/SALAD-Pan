# Configs

This directory contains all runtime configuration files used by training and inference scripts.

## Files

- `accelerate.yaml`: multi-GPU / mixed-precision launcher settings for `accelerate`.
- `train_vae.yaml`: Stage-I 1-channel VAE training config (`train_vae.py`).
- `train_diffusion.yaml`: Stage-II diffusion + dual-branch adapter training config (`train_diffusion.py`).
- `inference.yaml`: offline inference / evaluation config (`inference.py`).

## Naming Notes

The project now uses neutral naming in configs:

- `unet_adapter_model_name_or_path`
- `adapter_size_ratio`
- `adapter_learn_time_embedding`
- `adapter_time_embedding_mix`
- `conditioning_scale_spa`
- `conditioning_scale_spe`

Legacy keys are still accepted by scripts via internal compatibility mapping.

## Quick Usage

```bash
accelerate launch --config_file configs/accelerate.yaml train_vae.py --config configs/train_vae.yaml
accelerate launch --config_file configs/accelerate.yaml train_diffusion.py --config configs/train_diffusion.yaml
python inference.py --config configs/inference.yaml
```

## CLI Override

All three scripts support `-o key=value` overrides, for example:

```bash
python inference.py --config configs/inference.yaml -o inference_count=8 -o save_visual_rgb=false
```
