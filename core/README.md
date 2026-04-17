# Core

This directory contains the project-specific model and pipeline implementations used by training and inference.

## Structure

- `components/salad_pan.py`
  - Defines the dual-branch adapter and fused UNet model.
  - Main classes: `DualBranchXSAdapter`, `UNetDualBranchXSModel`.
- `pipelines/salad_pan.py`
  - Defines `StableDiffusionDualBranchXSPipeline`.
  - Supports direct conditioning input (`[B, 5, H, W]`).

## Design Intent

- Keep the lower-level model interface compatible with Diffusers ecosystem patterns.
- Keep training/inference entry scripts on neutral naming.
- Avoid breaking existing serialized weights and checkpoints.
