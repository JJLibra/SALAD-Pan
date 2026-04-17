"""
Calibrate scaling_factor for a trained 1-channel VAE using 4-channel H5 remote sensing data.

This script:
1. Loads 4-channel HRMS images from an H5 file.
2. Normalizes them to [-1, 1] using the same preprocessing as VAE training.
3. Splits each 4-channel image into 4 single-channel frames.
4. Encodes them with a trained 1-channel VAE.
5. Estimates the latent standard deviation E[std(z_raw)].
6. Recommends a new scaling_factor so that latent std is close to a target value
   (typically 1.0).

Typical use case:
- You trained a 1-channel VAE, but your real data is 4-channel multispectral imagery.
- You want to reuse the 1-channel VAE by treating each spectral band as an independent frame.
- Before diffusion training, you want a better scaling_factor than the default SD value.

Example command:
python tools/calibrate_vae_scaling_factor.py \
  --vae_path output/vae_c1_gf2_qb_wv3 \
  --h5_path data/gf2/train_gf2.h5 \
  --h5_key gt \
  --resolution 256 \
  --clip_min 0 \
  --clip_max 1023 \
  --batch_size 16 \
  --max_samples 2048 \
  --max_batches 128 \
  --target_latent_std 1.0 \
  --mixed_precision fp16
"""

import argparse
import contextlib
import random
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from tqdm.auto import tqdm


class H5VaeCalibrationDataset(torch.utils.data.Dataset):
    """
    Dataset for loading 4-channel GT images from an H5 file.

    Expected H5 structure:
        /gt : (N, 4, H, W)

    Preprocessing:
      - clip to [clip_min, clip_max]
      - divide by clip_max -> [0, 1]
      - map to [-1, 1]
      - resize to `resolution` if needed

    Returns:
        {
            "pixel_values": Tensor of shape [4, H, W], range [-1, 1]
        }
    """

    def __init__(
        self,
        h5_path: str,
        key: str = "gt",
        resolution: int = 256,
        clip_min: float = 0.0,
        clip_max: float = 1023.0,
        discard_out_of_range: bool = False,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.key = str(key)
        self.resolution = int(resolution)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.discard_out_of_range = bool(discard_out_of_range)

        with h5py.File(self.h5_path, "r") as f:
            if self.key not in f:
                raise KeyError(f"H5 key '{self.key}' not found in: {self.h5_path}")

            gt = f[self.key]
            if gt.ndim != 4:
                raise ValueError(f"Expected dataset shape (N,4,H,W), got {gt.shape}")
            if gt.shape[1] != 4:
                raise ValueError(f"Expected 4 channels, got shape {gt.shape}")

            num_samples_total = gt.shape[0]
            indices = list(range(num_samples_total))

            if self.discard_out_of_range:
                kept, discarded = [], 0
                for i in indices:
                    mn = float(gt[i].min())
                    mx = float(gt[i].max())
                    if (mn < self.clip_min) or (mx > self.clip_max):
                        discarded += 1
                    else:
                        kept.append(i)

                print(
                    f"[Dataset] total={num_samples_total}, kept={len(kept)}, "
                    f"discarded={discarded} (outside [{self.clip_min}, {self.clip_max}])"
                )
                self.indices = kept
            else:
                print(
                    f"[Dataset] total={num_samples_total}, kept={num_samples_total}, discarded=0 "
                    f"(outside [{self.clip_min}, {self.clip_max}])"
                )
                self.indices = indices

        if seed is not None:
            random.Random(seed).shuffle(self.indices)

        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

        self._h5 = None
        self._gt = None

    def _open_if_needed(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._gt = self._h5[self.key]

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _resize_if_needed(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) != tuple(size_hw):
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return x

    def __getitem__(self, idx: int):
        self._open_if_needed()
        real_idx = self.indices[idx]

        gt = np.array(self._gt[real_idx], dtype=np.float32)  # (4, H, W)
        gt = np.clip(gt, self.clip_min, self.clip_max) / self.clip_max
        gt = gt * 2.0 - 1.0  # -> [-1, 1]

        h, w = gt.shape[-2:]
        if (h != self.resolution) or (w != self.resolution):
            gt_t = torch.from_numpy(gt[None])  # [1, 4, H, W]
            gt_t = self._resize_if_needed(gt_t, (self.resolution, self.resolution))
            gt = gt_t[0].numpy()

        return {"pixel_values": torch.from_numpy(gt).to(torch.float32)}


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values}


def pack_4ch_to_1ch_frames(x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    Convert a batch of 4-channel images into independent 1-channel frames.

    Input:
        x: [B, 4, H, W]

    Output:
        x_frames: [B * 4, 1, H, W]
        and original shape metadata (B, C, H, W)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x.shape=[B,4,H,W], got {tuple(x.shape)}")

    b, c, h, w = x.shape
    if c != 4:
        raise ValueError(f"Expected 4 channels, got {c}")

    x_frames = x.view(b * c, 1, h, w)
    return x_frames, b, c, h, w


@torch.no_grad()
def estimate_latent_std(
    vae: AutoencoderKL,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    weight_dtype: torch.dtype,
    max_batches: Optional[int] = None,
    sample_posterior: bool = False,
) -> float:
    """
    Estimate E[std(z_raw)] over a subset of batches.

    Args:
        vae: trained 1-channel VAE
        dataloader: calibration dataloader
        device: torch device
        weight_dtype: dtype used during encoding
        max_batches: maximum number of batches to process; None = no limit
        sample_posterior: if True, use posterior.sample(); else posterior.mode()

    Returns:
        mean_std: mean of batch-wise latent std values
    """
    vae.eval()
    use_autocast = device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)

    std_values = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Estimating latent std")):
        x4 = batch["pixel_values"].to(device=device, dtype=weight_dtype)   # [B,4,H,W]
        x_frames, _, _, _, _ = pack_4ch_to_1ch_frames(x4)                  # [B*4,1,H,W]

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=weight_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            posterior = vae.encode(x_frames).latent_dist
            z_raw = posterior.sample() if sample_posterior else posterior.mode()

        z_cpu = z_raw.float().cpu()
        std_values.append(z_cpu.std().item())

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    if len(std_values) == 0:
        raise RuntimeError("No batches were processed. Please check your H5 path, key, and filtering settings.")

    return float(np.mean(std_values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate latent std and recommend scaling_factor for a trained 1-channel VAE on 4-channel H5 data."
    )

    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to a trained 1-channel VAE directory saved with `save_pretrained()`.",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to an H5 file containing 4-channel GT data.",
    )
    parser.add_argument(
        "--h5_key",
        type=str,
        default="gt",
        help="Dataset key in the H5 file. Default: gt",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Target resolution for resizing. Should match VAE training resolution.",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.0,
        help="Lower clipping bound.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=1023.0,
        help="Upper clipping bound.",
    )
    parser.add_argument(
        "--discard_out_of_range",
        action="store_true",
        help="Discard samples whose values fall outside [clip_min, clip_max].",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2048,
        help="Maximum number of 4-channel images to use. Use -1 to process all samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Dataloader batch size.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=128,
        help="Maximum number of batches to process. Use -1 for no limit.",
    )
    parser.add_argument(
        "--target_latent_std",
        type=float,
        default=1.0,
        help="Target latent standard deviation. Usually 1.0.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for evaluation on CUDA.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU mode even if CUDA is available.",
    )
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="Use posterior.sample() instead of posterior.mode() when estimating latent std.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for dataset shuffling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Dtype
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    print(f"[Info] Device: {device}")
    print(f"[Info] Weight dtype: {weight_dtype}")

    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE path not found: {vae_path}")

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 path not found: {h5_path}")

    max_samples = None if (args.max_samples is not None and args.max_samples < 0) else args.max_samples
    max_batches = None if (args.max_batches is not None and args.max_batches < 0) else args.max_batches

    dataset = H5VaeCalibrationDataset(
        h5_path=str(h5_path),
        key=args.h5_key,
        resolution=args.resolution,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        discard_out_of_range=args.discard_out_of_range,
        max_samples=max_samples,
        seed=args.seed,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Info] Loading 1-channel VAE from: {vae_path}")
    vae = AutoencoderKL.from_pretrained(str(vae_path))
    vae.to(device=device, dtype=weight_dtype)

    old_scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    print(f"[Info] Current scaling_factor in config: {old_scaling_factor:.6f}")

    mean_std = estimate_latent_std(
        vae=vae,
        dataloader=dataloader,
        device=device,
        weight_dtype=weight_dtype,
        max_batches=max_batches,
        sample_posterior=args.sample_posterior,
    )

    target_std = float(args.target_latent_std)
    eps = 1e-8
    recommended_scaling_factor = target_std / max(mean_std, eps)

    print("\n========== VAE scaling_factor calibration ==========")
    print(f"Estimated E[std(z_raw)]      : {mean_std:.6f}")
    print(f"Target latent std            : {target_std:.6f}")
    print(f"Current scaling_factor       : {old_scaling_factor:.6f}")
    print(f"Recommended scaling_factor   : {recommended_scaling_factor:.6f}")
    print(f"Ratio new / old              : {recommended_scaling_factor / old_scaling_factor:.6f}")
    print("====================================================")

    print("\nTo update your VAE config manually:")
    print("  from diffusers import AutoencoderKL")
    print(f"  vae = AutoencoderKL.from_pretrained('{vae_path}')")
    print(f"  vae.config.scaling_factor = {recommended_scaling_factor:.6f}")
    print("  vae.save_pretrained('path/to/new_or_same_directory')")


if __name__ == "__main__":
    main()
