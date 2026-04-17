# Data

The project uses datasets from [PanCollection](https://github.com/liangjiandeng/PanCollection).

## Directory Layout

Recommended structure:

```text
data/
  gf2/
  qb/
  wv3/
  wv2/
  vae/
```

## H5 Conventions

Training and inference scripts expect H5 datasets with keys configured in YAML (default):

- `gt`: ground-truth HRMS, shape `[N, C, H, W]`
- `lms`: low-resolution multispectral, shape `[N, C, H, W]`
- `pan`: panchromatic, shape `[N, 1, H, W]`

For Stage-I VAE training, a single-channel `gt` H5 is expected.

## Important

- Keep value ranges consistent with config (`range_clip_min`, `range_clip_max`, `range_clip_max_map`).
- If datasets have different bit-depth/ranges (e.g., GF2 vs QB/WV3), set per-dataset clip max in YAML.
