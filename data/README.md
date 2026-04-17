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
    train_gt_1ch_all.h5
```

## H5 Conventions

Stage-II diffusion/inference H5 files (default keys in YAML):

- `gt`: ground-truth HRMS, shape `[N, C, H, W]`
- `lms`: low-resolution multispectral, shape `[N, C, H, W]`
- `pan`: panchromatic, shape `[N, 1, H, W]`

Stage-I VAE training uses a single-channel GT file:

- `data/vae/train_gt_1ch_all.h5`
- required dataset: `gt` with shape `[N, 1, H, W]`

## Build `train_gt_1ch_all.h5`

`utils/get_training_vae_data.py` merges GF2/QB/WV3 `gt` into one 1-channel dataset.

Example:

```bash
python utils/get_training_vae_data.py \
  --gf2 data/gf2/train_gf2.h5 \
  --qb  data/qb/train_qb.h5 \
  --wv3 data/wv3/train_wv3.h5 \
  --out data/vae/train_gt_1ch_all.h5 \
  --shuffle --seed 2025
```

What it does:

1. Reads each input file's `gt` with shape `[N, C, H, W]`.
2. Checks all inputs have the same spatial size `H, W`.
3. Expands each `(image, band)` into one 1-channel sample.
4. Writes merged output `gt` as `[sum_i(N_i*C_i), 1, H, W]`.
5. Also writes trace metadata:
   - `sensor_id`
   - `img_index`
   - `band_index`
   - `sensor_name`

## Compatibility With `train_vae.py`

- `configs/train_vae.yaml` uses:
  - `train_h5_path: data/vae/train_gt_1ch_all.h5`
  - `h5_keys.gt: gt`
- `train_vae.py` only requires `gt`; extra metadata datasets are ignored safely.

## Important

- Keep value ranges consistent with config (`range_clip_min`, `range_clip_max`, `range_clip_max_map`).
- If datasets have different bit-depth/ranges (e.g., GF2 vs QB/WV3), set per-dataset clip max in YAML.
