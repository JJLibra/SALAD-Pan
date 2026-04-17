# Checkpoints

This directory stores reusable pretrained weights and released model snapshots.

## Suggested Layout

```text
checkpoints/
  vae_c1/
    ... (Stage-I VAE checkpoints)
  diffusion/
    ... (Stage-II diffusion/adapter checkpoints)
```

## What to Put Here

- Stage-I VAE weights (for `vae_path` in configs).
- Stage-II adapter or full model checkpoints.

## Naming Tip

For Stage-II adapter-only exports, use a clear filename such as:

- `dual_branch_xs_adapter.pt`

Then point `adapter_weights_path` in `configs/inference.yaml` to that file.
