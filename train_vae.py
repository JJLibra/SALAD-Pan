import argparse
import ast
import json
import logging
import math
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from utils.metrics import Q4_numpy, Q8_numpy, SAM_numpy, ERGAS_numpy, SCC_numpy

check_min_version("0.36.0.dev0")
logger = get_logger(__name__)


def _infer_q_metric_from_tag(tag: str):
    return (Q8_numpy, "q8") if str(tag).lower() == "wv3" else (Q4_numpy, "q4")


def _build_validation_targets(args, tag_prefix: str):
    targets = []

    # Preferred generic spec list:
    # validation_specs:
    #   - name: gf2
    #     h5_path: ...
    #     clip_min: 0
    #     clip_max: 2047
    #     q_metric: q4|q8
    raw_specs = getattr(args, "validation_specs", None)
    if isinstance(raw_specs, list) and len(raw_specs) > 0:
        for spec in raw_specs:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name", "custom")).strip() or "custom"
            h5_path = spec.get("h5_path", None)
            if not h5_path:
                continue
            clip_min = float(spec.get("clip_min", args.range_clip_min))
            clip_max = float(spec.get("clip_max", args.range_clip_max))
            q_metric = str(spec.get("q_metric", "")).strip().lower()
            if q_metric == "q8":
                q_metric_fn, q_tag_name = Q8_numpy, "q8"
            elif q_metric == "q4":
                q_metric_fn, q_tag_name = Q4_numpy, "q4"
            else:
                q_metric_fn, q_tag_name = _infer_q_metric_from_tag(name)
            targets.append(
                {
                    "tag_prefix": f"{tag_prefix}_{name}",
                    "h5_path": h5_path,
                    "clip_min": clip_min,
                    "clip_max": clip_max,
                    "q_metric_fn": q_metric_fn,
                    "q_tag_name": q_tag_name,
                }
            )
        if len(targets) > 0:
            return targets

    # Backward-compatible legacy keys
    for sensor_name in ("gf2", "qb", "wv3"):
        h5_path = getattr(args, f"validation_h5_path_{sensor_name}", None)
        if not h5_path:
            continue
        q_metric_fn, q_tag_name = _infer_q_metric_from_tag(sensor_name)
        targets.append(
            {
                "tag_prefix": f"{tag_prefix}_{sensor_name}",
                "h5_path": h5_path,
                "clip_min": float(getattr(args, f"range_clip_min_{sensor_name}")),
                "clip_max": float(getattr(args, f"range_clip_max_{sensor_name}")),
                "q_metric_fn": q_metric_fn,
                "q_tag_name": q_tag_name,
            }
        )

    if len(targets) == 0 and getattr(args, "validation_h5_path", None):
        targets.append(
            {
                "tag_prefix": tag_prefix,
                "h5_path": args.validation_h5_path,
                "clip_min": float(args.range_clip_min),
                "clip_max": float(args.range_clip_max),
                "q_metric_fn": Q4_numpy,
                "q_tag_name": "q4",
            }
        )

    return targets


def _literal(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_image_safe(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_validation_rgb_pair(base_dir: Path, tag_prefix: str, sample_idx: int, dataset_idx: int, gt_rgb, rc_rgb):
    sample_dir = base_dir / tag_prefix / f"{sample_idx:04d}_idx{dataset_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image_safe(gt_rgb, sample_dir / "gt_rgb.png")
    save_image_safe(rc_rgb, sample_dir / "recon_rgb.png")


def _require_keys(cfg: dict, required_keys):
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(
            "Missing required config keys:\n" + "\n".join([f"  - {k}" for k in missing])
        )


def load_config():
    ap = argparse.ArgumentParser(
        description="Strict-config 1ch VAE training from single-channel H5 + multi-spectral validation",
        add_help=True,
    )
    ap.add_argument("--config", type=str, required=True, help="YAML config file path")
    ap.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Override some fields in YAML, e.g.: -o train_batch_size=8 -o learning_rate=1e-4",
    )
    cli, unknown = ap.parse_known_args()

    if unknown:
        print(
            f"[WARN] Ignoring unused CLI args: {unknown}. "
            f"This script only accepts --config and -o/--override."
        )

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cli.config} did not parse into a dict.")

    for item in cli.override:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item} (should be key=value)")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _literal(v.strip())

    required_keys = [
        "pretrained_model_name_or_path",
        "output_dir",
        "logging_dir",
        "resolution",
        "seed",
        "local_files_only",
        "train_h5_path",
        "h5_keys",
        "range_clip_min",
        "range_clip_max",
        "discard_out_of_range",
        "max_train_samples",
        "train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_train_steps",
        "dataloader_num_workers",
        "mixed_precision",
        "allow_tf32",
        "scale_lr",
        "use_8bit_adam",
        "learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_weight_decay",
        "adam_epsilon",
        "max_grad_norm",
        "set_grads_to_none",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_num_cycles",
        "lr_power",
        "checkpointing_steps",
        "checkpoints_total_limit",
        "train_first_last_only",
        "sample_posterior",
        "lambda_charbonnier",
        "charbonnier_eps",
        "lambda_ssim",
        "ssim_kernel",
        "ssim_sigma",
        "ssim_pool",
        "lambda_mse",
        "lambda_mae",
        "lambda_psnr",
        "calibrate_scaling_factor",
        "target_latent_std",
        "calib_num_samples",
        "calib_batch_size",
        "scaling_factor_override",
        "validation_steps",
        "save_validation_rgb",
        "validation_fixed_first_n",
        "validation_random_count",
        "validation_h5_path",
        "validation_h5_path_gf2",
        "validation_h5_path_qb",
        "validation_h5_path_wv3",
        "range_clip_min_gf2",
        "range_clip_max_gf2",
        "range_clip_min_qb",
        "range_clip_max_qb",
        "range_clip_min_wv3",
        "range_clip_max_wv3",
    ]
    _require_keys(cfg, required_keys)

    # Optional generic validation specs (preferred), kept backward compatible with legacy keys.
    if "validation_specs" not in cfg:
        cfg["validation_specs"] = None

    if cfg["resolution"] % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8.")

    train_h5_path = cfg["train_h5_path"]
    if not train_h5_path:
        raise ValueError("`train_h5_path` must be provided.")
    if not Path(train_h5_path).exists():
        raise FileNotFoundError(f"Train H5 file not found: {train_h5_path}")

    if not isinstance(cfg["h5_keys"], dict):
        raise ValueError("`h5_keys` must be a dict, e.g. {'gt': 'gt'}")
    if "gt" not in cfg["h5_keys"]:
        raise KeyError("`h5_keys` must contain key `gt`.")

    return SimpleNamespace(**cfg)


class H5VaeDataset(torch.utils.data.Dataset):
    """
    Train set:
      gt: (N,1,H,W) or (N,H,W)
      preprocess:
        clip -> [clip_min, clip_max]
        /clip_max -> [0,1]
        -> [-1,1]
    """
    def __init__(
        self,
        h5_path: str,
        key: str,
        resolution: int,
        clip_min: float,
        clip_max: float,
        discard_out_of_range: bool,
        max_train_samples,
        seed,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.key = key
        self.resolution = int(resolution)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.discard_out = bool(discard_out_of_range)

        with h5py.File(self.h5_path, "r") as f:
            gt = f[self.key]
            N = gt.shape[0]
            indices = list(range(N))

            if self.discard_out:
                keep, out_cnt = [], 0
                for i in indices:
                    mn = float(gt[i].min())
                    mx = float(gt[i].max())
                    if (mn < self.clip_min) or (mx > self.clip_max):
                        out_cnt += 1
                    else:
                        keep.append(i)
                logger.info(
                    f"[H5Dataset-Train1ch] {h5_path} total={N}, kept={len(keep)}, "
                    f"discarded={out_cnt} (outside [{self.clip_min},{self.clip_max}])"
                )
                self.indices = keep
            else:
                logger.info(
                    f"[H5Dataset-Train1ch] {h5_path} total={N}, kept={N}, discarded=0 "
                    f"(outside [{self.clip_min},{self.clip_max}])"
                )
                self.indices = indices

        if seed is not None:
            random.Random(seed).shuffle(self.indices)
        if max_train_samples is not None:
            self.indices = self.indices[: int(max_train_samples)]

        self._h5 = None
        self._gt = None

    def _open_if_needed(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._gt = self._h5[self.key]

    def __len__(self):
        return len(self.indices)

    def _resize_if_needed(self, x: torch.Tensor, size_hw):
        if tuple(x.shape[-2:]) != tuple(size_hw):
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return x

    def __getitem__(self, idx):
        self._open_if_needed()
        ridx = self.indices[idx]
        gt = np.array(self._gt[ridx], dtype=np.float32)

        if gt.ndim == 2:
            gt = gt[None, ...]
        elif gt.ndim == 3 and gt.shape[0] != 1:
            raise ValueError(f"Train H5 expected 1 channel, got shape {gt.shape}")

        gt = np.clip(gt, self.clip_min, self.clip_max) / self.clip_max
        gt = gt * 2.0 - 1.0

        H, W = gt.shape[-2:]
        if (H != self.resolution) or (W != self.resolution):
            gt_t = torch.from_numpy(gt[None])
            gt_t = self._resize_if_needed(gt_t, (self.resolution, self.resolution))
            gt = gt_t[0].numpy()

        return {"pixel_values": torch.from_numpy(gt).to(torch.float32)}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values}


def calc_mse(x, y):
    return F.mse_loss(x, y, reduction="mean")


def calc_mae(x, y):
    return F.l1_loss(x, y, reduction="mean")


def calc_psnr(x01, y01, eps=1e-10):
    mse = calc_mse(x01, y01).clamp(min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred, target, eps):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def _gauss_1d(ks, sigma, device, dtype):
    xs = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    w = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    return (w / w.sum()).unsqueeze(0)


def _ssim_kernel(ks, sigma, c, device, dtype):
    g = _gauss_1d(ks, sigma, device, dtype)
    g2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return g2d.repeat(c, 1, 1, 1)


def ssim_loss(x01, y01, ks, sigma, pool):
    if pool > 1:
        x01 = F.avg_pool2d(x01, kernel_size=pool)
        y01 = F.avg_pool2d(y01, kernel_size=pool)

    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    _, c, _, _ = x01.shape
    k = _ssim_kernel(ks, sigma, c, x01.device, x01.dtype)
    pad = ks // 2
    mu_x = F.conv2d(x01, k, groups=c, padding=pad)
    mu_y = F.conv2d(y01, k, groups=c, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = F.conv2d(x01 * x01, k, groups=c, padding=pad) - mu_x2
    sig_y2 = F.conv2d(y01 * y01, k, groups=c, padding=pad) - mu_y2
    sig_xy = F.conv2d(x01 * y01, k, groups=c, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return 1.0 - ssim_map.mean()


def _pack_ms_to_1ch_frames(x: torch.Tensor):
    if x.ndim != 4:
        raise ValueError(f"Expected x.shape=[B,C,H,W], got {x.shape}")
    B, C, H, W = x.shape
    return x.view(B * C, 1, H, W), B, C, H, W


def _unpack_1ch_frames_to_ms(y_frames: torch.Tensor, B: int, C: int):
    if y_frames.ndim != 4:
        raise ValueError(f"Expected y_frames.shape=[B*C,1,H,W], got {y_frames.shape}")
    _, one, H, W = y_frames.shape
    if one != 1:
        raise ValueError(f"Expected single-channel frames, got {one}")
    return y_frames.view(B, C, H, W)


@torch.no_grad()
def build_1ch_vae_from_3ch(sd_root_or_vae_path: str, local_files_only: bool):
    try:
        base_vae = AutoencoderKL.from_pretrained(
            sd_root_or_vae_path, subfolder="vae", local_files_only=local_files_only
        )
        logger.info(f"[VAE load] loaded with subfolder='vae' from: {sd_root_or_vae_path}")
    except EnvironmentError:
        base_vae = AutoencoderKL.from_pretrained(sd_root_or_vae_path, local_files_only=local_files_only)
        logger.info(f"[VAE load] loaded directly from: {sd_root_or_vae_path}")

    base_in_cfg = getattr(base_vae.config, "in_channels", None)
    base_out_cfg = getattr(base_vae.config, "out_channels", None)

    sd = base_vae.state_dict()
    w_in = sd.get("encoder.conv_in.weight", None)
    w_out = sd.get("decoder.conv_out.weight", None)

    base_in_w = int(w_in.shape[1]) if w_in is not None else None
    base_out_w = int(w_out.shape[0]) if w_out is not None else None

    is_1ch = False
    if base_in_cfg is not None and base_out_cfg is not None:
        is_1ch = (int(base_in_cfg) == 1 and int(base_out_cfg) == 1)
    if (not is_1ch) and (base_in_w is not None and base_out_w is not None):
        is_1ch = (base_in_w == 1 and base_out_w == 1)

    if is_1ch:
        logger.info(
            f"[Load 1ch VAE directly] in_channels={base_in_cfg or base_in_w}, "
            f"out_channels={base_out_cfg or base_out_w}"
        )
        return base_vae

    if w_in is None or w_out is None:
        raise ValueError(
            "Cannot find encoder.conv_in.weight or decoder.conv_out.weight in the loaded VAE state_dict."
        )

    _, c_in, _, _ = w_in.shape
    if c_in != 3:
        raise ValueError(
            f"Expected a 3ch VAE for conversion, but encoder.conv_in.weight Cin={c_in}."
        )

    cout2, _, _, _ = w_out.shape
    if cout2 != 3:
        raise ValueError(
            f"Expected a 3ch VAE for conversion, but decoder.conv_out.weight Cout={cout2}."
        )

    cfg_dict = base_vae.config.to_dict() if hasattr(base_vae.config, "to_dict") else dict(base_vae.config)
    cfg_dict["in_channels"] = 1
    cfg_dict["out_channels"] = 1
    vae1 = AutoencoderKL.from_config(cfg_dict)

    lum = torch.tensor([0.2989, 0.5870, 0.1140], dtype=w_in.dtype, device=w_in.device)
    sd["encoder.conv_in.weight"] = (w_in * lum.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    b_out = sd.get("decoder.conv_out.bias", None)
    sd["decoder.conv_out.weight"] = (w_out * lum.view(3, 1, 1, 1)).sum(dim=0, keepdim=True)
    if b_out is not None:
        sd["decoder.conv_out.bias"] = (b_out * lum).sum(dim=0, keepdim=True)

    missing, unexpected = vae1.load_state_dict(sd, strict=False)
    logger.info(f"[Compat load 3ch->1ch] missing={len(missing)}, unexpected={len(unexpected)}")

    vae1.config.scaling_factor = float(getattr(base_vae.config, "scaling_factor", 0.18215))
    return vae1


@torch.no_grad()
def _log_validation_one_sensor(
    vae,
    args,
    accelerator,
    weight_dtype,
    step,
    tag_prefix,
    h5_path,
    clip_min,
    clip_max,
    q_metric_fn,
    q_tag_name,
):
    if not h5_path:
        return
    if not accelerator.is_main_process:
        return

    raw_vae = accelerator.unwrap_model(vae)
    was_training = raw_vae.training
    raw_vae.eval()

    key = args.h5_keys["gt"]
    res = int(args.resolution)
    n_fixed = int(args.validation_fixed_first_n)
    n_rand_target = int(args.validation_random_count)

    rng = np.random.default_rng(args.seed)
    val_vis_dir = Path(args.output_dir) / "validation_vis"
    val_metric_file = Path(args.output_dir) / "validation_metrics.jsonl"

    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        N = ds.shape[0]
        if N == 0:
            logger.warning(f"[Validation] Empty validation set: {h5_path}")
            return

        fixed_idxs = list(range(min(n_fixed, N)))

        all_idxs = np.arange(N)
        remain_pool = np.setdiff1d(all_idxs, fixed_idxs, assume_unique=True)
        if len(remain_pool) > 0 and n_rand_target > 0:
            rand_size = min(n_rand_target, len(remain_pool))
            rand_idxs = rng.choice(remain_pool, size=rand_size, replace=False).tolist()
        else:
            rand_idxs = []

        for i, idx in enumerate(fixed_idxs):
            arr = np.array(ds[idx], dtype=np.float32)
            arr = np.clip(arr, clip_min, clip_max) / clip_max
            arr = arr * 2.0 - 1.0
            t = torch.from_numpy(arr)[None]

            if t.shape[-1] != res or t.shape[-2] != res:
                t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)

            x = t.to(device=accelerator.device, dtype=weight_dtype)
            x_frames, B, C, _, _ = _pack_ms_to_1ch_frames(x)

            with torch.autocast("cuda", dtype=weight_dtype if accelerator.device.type == "cuda" else None):
                posterior = raw_vae.encode(x_frames).latent_dist
                z = posterior.sample() if args.sample_posterior else posterior.mode()
                recon_frames = raw_vae.decode(z).sample

            recon = _unpack_1ch_frames_to_ms(recon_frames, B, C)

            x01 = ((x.float() + 1.0) * 0.5).clamp(0, 1)
            recon01 = ((recon.float() + 1.0) * 0.5).clamp(0, 1)

            psnr_i = calc_psnr(recon01, x01).item()
            ssim_i = 1.0 - ssim_loss(
                recon01, x01,
                ks=int(args.ssim_kernel),
                sigma=float(args.ssim_sigma),
                pool=int(args.ssim_pool),
            ).item()

            gen_np = recon01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            gt_np = x01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            try:
                q_i = float(q_metric_fn(gt_np, gen_np))
            except Exception as e:
                logger.warning(f"{q_tag_name.upper()} computation failed at idx={idx}: {e}")
                q_i = float("nan")
            try:
                sam_i = float(SAM_numpy(gt_np, gen_np))
            except Exception as e:
                logger.warning(f"SAM computation failed at idx={idx}: {e}")
                sam_i = float("nan")
            try:
                ergas_i = float(ERGAS_numpy(gt_np, gen_np))
            except Exception as e:
                logger.warning(f"ERGAS computation failed at idx={idx}: {e}")
                ergas_i = float("nan")
            try:
                scc_i = float(SCC_numpy(gt_np, gen_np))
            except Exception as e:
                logger.warning(f"SCC computation failed at idx={idx}: {e}")
                scc_i = float("nan")

            if args.save_validation_rgb:
                try:
                    if x01.shape[1] >= 3:
                        x01_rgb = x01[:, 1:4]
                        recon01_rgb = recon01[:, 1:4]
                        gt_rgb = to_pil_image(x01_rgb[0].cpu())
                        rc_rgb = to_pil_image(recon01_rgb[0].cpu())
                        save_validation_rgb_pair(val_vis_dir, tag_prefix, i, idx, gt_rgb, rc_rgb)
                except Exception as e:
                    logger.warning(f"{tag_prefix} image saving failed at idx={idx}: {e}")

            append_jsonl(
                val_metric_file,
                {
                    "time": datetime.now().isoformat(),
                    "step": int(step),
                    "tag_prefix": tag_prefix,
                    "sample_kind": "fixed",
                    "sample_order": int(i),
                    "dataset_idx": int(idx),
                    "psnr": float(psnr_i),
                    "ssim": float(ssim_i),
                    q_tag_name: float(q_i),
                    "sam": float(sam_i),
                    "ergas": float(ergas_i),
                    "scc": float(scc_i),
                },
            )

        rand_psnr, rand_ssim, rand_q, rand_sam, rand_ergas, rand_scc = [], [], [], [], [], []

        for idx in rand_idxs:
            arr = np.array(ds[idx], dtype=np.float32)
            arr = np.clip(arr, clip_min, clip_max) / clip_max
            arr = arr * 2.0 - 1.0
            t = torch.from_numpy(arr)[None]

            if t.shape[-1] != res or t.shape[-2] != res:
                t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)

            x = t.to(device=accelerator.device, dtype=weight_dtype)
            x_frames, B, C, _, _ = _pack_ms_to_1ch_frames(x)

            with torch.autocast("cuda", dtype=weight_dtype if accelerator.device.type == "cuda" else None):
                posterior = raw_vae.encode(x_frames).latent_dist
                z = posterior.mode()
                recon_frames = raw_vae.decode(z).sample

            recon = _unpack_1ch_frames_to_ms(recon_frames, B, C)

            x01 = ((x.float() + 1.0) * 0.5).clamp(0, 1)
            recon01 = ((recon.float() + 1.0) * 0.5).clamp(0, 1)

            rand_psnr.append(calc_psnr(recon01, x01).item())
            rand_ssim.append(
                1.0 - ssim_loss(
                    recon01, x01,
                    ks=int(args.ssim_kernel),
                    sigma=float(args.ssim_sigma),
                    pool=int(args.ssim_pool),
                ).item()
            )

            gen_np = recon01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            gt_np = x01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            try:
                rand_q.append(float(q_metric_fn(gt_np, gen_np)))
            except Exception as e:
                logger.warning(f"{q_tag_name.upper()} computation failed at idx={idx}: {e}")
                rand_q.append(float("nan"))
            try:
                rand_sam.append(float(SAM_numpy(gt_np, gen_np)))
            except Exception as e:
                logger.warning(f"SAM computation failed at idx={idx}: {e}")
                rand_sam.append(float("nan"))
            try:
                rand_ergas.append(float(ERGAS_numpy(gt_np, gen_np)))
            except Exception as e:
                logger.warning(f"ERGAS computation failed at idx={idx}: {e}")
                rand_ergas.append(float("nan"))
            try:
                rand_scc.append(float(SCC_numpy(gt_np, gen_np)))
            except Exception as e:
                logger.warning(f"SCC computation failed at idx={idx}: {e}")
                rand_scc.append(float("nan"))

        if len(rand_psnr) > 0:
            append_jsonl(
                val_metric_file,
                {
                    "time": datetime.now().isoformat(),
                    "step": int(step),
                    "tag_prefix": tag_prefix,
                    "sample_kind": f"rand{len(rand_psnr)}",
                    "mean_psnr": float(np.nanmean(rand_psnr)),
                    "mean_ssim": float(np.nanmean(rand_ssim)),
                    f"mean_{q_tag_name}": float(np.nanmean(rand_q)),
                    "mean_sam": float(np.nanmean(rand_sam)),
                    "mean_ergas": float(np.nanmean(rand_ergas)),
                    "mean_scc": float(np.nanmean(rand_scc)),
                },
            )

    if was_training:
        raw_vae.train()


@torch.no_grad()
def log_validation_h5(vae, args, accelerator, weight_dtype, step, tag_prefix="val_h5"):
    targets = _build_validation_targets(args, tag_prefix=tag_prefix)
    if len(targets) == 0:
        return

    for t in targets:
        _log_validation_one_sensor(
            vae=vae,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=step,
            tag_prefix=t["tag_prefix"],
            h5_path=t["h5_path"],
            clip_min=t["clip_min"],
            clip_max=t["clip_max"],
            q_metric_fn=t["q_metric_fn"],
            q_tag_name=t["q_tag_name"],
        )


@torch.no_grad()
def calibrate_scaling_factor(vae, dataset, args, accelerator, weight_dtype):
    if not args.calibrate_scaling_factor:
        return

    raw_vae = accelerator.unwrap_model(vae)
    was_training = raw_vae.training
    raw_vae.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.calib_batch_size),
        shuffle=True,
        num_workers=0,
    )

    std_list = []
    num_batches = 0

    for batch in loader:
        x = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

        with torch.autocast("cuda", dtype=weight_dtype if accelerator.device.type == "cuda" else None):
            posterior = raw_vae.encode(x).latent_dist
            z_raw = posterior.sample()

        z_g = accelerator.gather(z_raw).float().cpu()
        std_list.append(z_g.std().item())

        num_batches += 1
        if num_batches >= int(args.calib_num_samples):
            break

    if len(std_list) > 0:
        mean_std = float(np.mean(std_list))
        old_sf = float(getattr(raw_vae.config, "scaling_factor", 0.18215))
        new_sf = float(args.target_latent_std) / max(mean_std, 1e-8)
        raw_vae.config.scaling_factor = new_sf
        logger.info(
            f"[Calibrate scaling_factor] old={old_sf:.6f}, std(z_raw)~{mean_std:.6f}, "
            f"target={float(args.target_latent_std):.6f} => new={new_sf:.6f}"
        )
    else:
        logger.warning("[Calibrate scaling_factor] No samples collected, scaling_factor unchanged.")

    if was_training:
        raw_vae.train()


def _parse_step_from_name(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def enforce_checkpoint_limit(output_dir: str, total_limit: int, prefix: str = "checkpoint"):
    if total_limit is None or total_limit <= 0:
        return
    root = Path(output_dir)
    if not root.exists():
        return
    ckpts = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{prefix}-")]
    if len(ckpts) <= total_limit:
        return
    ckpts_sorted = sorted(ckpts, key=_parse_step_from_name)
    to_delete = ckpts_sorted[: len(ckpts_sorted) - total_limit]
    for p in to_delete:
        try:
            shutil.rmtree(p, ignore_errors=True)
            logger.info(f"[Checkpoint GC] Removed old checkpoint: {p}")
        except Exception as e:
            logger.warning(f"[Checkpoint GC] Failed to remove {p}: {e}")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    def _as_float(x):
        return None if x is None else float(x)

    def _as_int(x):
        return None if x is None else int(x)

    def _as_bool(x):
        if isinstance(x, str):
            return x.lower() in ("1", "true", "yes", "y", "on")
        return bool(x)

    float_keys = [
        "learning_rate", "adam_beta1", "adam_beta2", "adam_weight_decay", "adam_epsilon",
        "range_clip_min", "range_clip_max",
        "range_clip_min_gf2", "range_clip_max_gf2",
        "range_clip_min_qb", "range_clip_max_qb",
        "range_clip_min_wv3", "range_clip_max_wv3",
        "target_latent_std", "scaling_factor_override",
        "lambda_charbonnier", "charbonnier_eps",
        "lambda_ssim", "ssim_sigma",
        "lambda_mse", "lambda_mae", "lambda_psnr",
        "lr_power", "max_grad_norm",
    ]
    int_keys = [
        "train_batch_size", "gradient_accumulation_steps", "lr_warmup_steps",
        "num_train_epochs", "max_train_steps", "resolution",
        "checkpointing_steps", "validation_steps",
        "calib_num_samples", "calib_batch_size", "dataloader_num_workers",
        "ssim_kernel", "ssim_pool", "checkpoints_total_limit",
        "validation_fixed_first_n", "validation_random_count", "lr_num_cycles",
    ]
    bool_keys = [
        "discard_out_of_range", "use_8bit_adam", "scale_lr", "train_first_last_only",
        "calibrate_scaling_factor", "sample_posterior", "local_files_only",
        "allow_tf32", "set_grads_to_none", "save_validation_rgb",
    ]

    for k in float_keys:
        setattr(args, k, _as_float(getattr(args, k)))
    for k in int_keys:
        setattr(args, k, _as_int(getattr(args, k)))
    for k in bool_keys:
        setattr(args, k, _as_bool(getattr(args, k)))

    if args.seed is not None:
        args.seed = int(args.seed)
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        with (Path(args.output_dir) / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=False)

    train_dataset = H5VaeDataset(
        h5_path=args.train_h5_path,
        key=args.h5_keys["gt"],
        resolution=args.resolution,
        clip_min=args.range_clip_min,
        clip_max=args.range_clip_max,
        discard_out_of_range=args.discard_out_of_range,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=bool(args.dataloader_num_workers > 0),
    )

    logger.info("Initializing 1ch VAE from pretrained VAE weights")
    vae = build_1ch_vae_from_3ch(
        args.pretrained_model_name_or_path,
        local_files_only=args.local_files_only,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    if args.scaling_factor_override is not None:
        vae.config.scaling_factor = float(args.scaling_factor_override)

    if args.train_first_last_only:
        vae.requires_grad_(False)
        for name, p in vae.named_parameters():
            if name.startswith("encoder.conv_in.") or name.startswith("decoder.conv_out."):
                p.requires_grad_(True)
        logger.info("[Trainable params] Only encoder.conv_in & decoder.conv_out are trainable.")
    else:
        vae.requires_grad_(True)
        logger.info("[Trainable params] All VAE params enabled.")

    vae, train_dataloader = accelerator.prepare(vae, train_dataloader)
    vae.train()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("bitsandbytes is required: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, accelerator.unwrap_model(vae).parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(
            f"[Auto] max_train_steps set to {args.max_train_steps} "
            f"(updates_per_epoch={num_update_steps_per_epoch}, epochs={args.num_train_epochs})"
        )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    num_training_steps_for_scheduler = int(args.max_train_steps * accelerator.num_processes * 1.05)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    calibrate_scaling_factor(vae, train_dataset, args, accelerator, weight_dtype)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    raw_vae = accelerator.unwrap_model(vae)
    train_metric_file = Path(args.output_dir) / "train_metrics.jsonl"

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(vae):
                x1 = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                with torch.autocast("cuda", dtype=weight_dtype if accelerator.device.type == "cuda" else None):
                    posterior = raw_vae.encode(x1).latent_dist
                    z = posterior.sample() if args.sample_posterior else posterior.mode()
                    recon1 = raw_vae.decode(z).sample

                x01 = ((x1.float() + 1.0) * 0.5).clamp(0, 1)
                recon01 = ((recon1.float() + 1.0) * 0.5).clamp(0, 1)

                L_char = charbonnier_loss(recon01, x01, eps=args.charbonnier_eps)
                L_ssim = ssim_loss(
                    recon01, x01,
                    ks=args.ssim_kernel,
                    sigma=args.ssim_sigma,
                    pool=args.ssim_pool,
                )
                L_mse = calc_mse(recon01, x01)
                L_mae = calc_mae(recon01, x01)
                psnr_val = calc_psnr(recon01, x01)
                L_psnr = 1.0 - psnr_val / 50.0

                loss = (
                    args.lambda_charbonnier * L_char
                    + args.lambda_ssim * L_ssim
                    + args.lambda_mse * L_mse
                    + args.lambda_mae * L_mae
                    + args.lambda_psnr * L_psnr
                ).float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(raw_vae.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.unwrap_model(vae).save_pretrained(save_path)
                    logger.info(f"Saved VAE to {save_path}")

                    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit > 0:
                        enforce_checkpoint_limit(args.output_dir, args.checkpoints_total_limit)

                has_val = (
                    args.validation_h5_path
                    or args.validation_h5_path_gf2
                    or args.validation_h5_path_qb
                    or args.validation_h5_path_wv3
                )
                if has_val and (global_step % args.validation_steps == 0):
                    accelerator.wait_for_everyone()
                    log_validation_h5(vae, args, accelerator, weight_dtype, step=global_step, tag_prefix="val_h5")
                    accelerator.wait_for_everyone()

            with torch.no_grad():
                psnr_1 = calc_psnr(recon01, x01).item()
                ssim_1 = 1.0 - ssim_loss(
                    recon01, x01,
                    ks=args.ssim_kernel,
                    sigma=args.ssim_sigma,
                    pool=args.ssim_pool,
                ).item()
                mse_1 = calc_mse(recon01, x01).item()
                mae_1 = calc_mae(recon01, x01).item()

            logs = {
                "time": datetime.now().isoformat(),
                "epoch": int(epoch),
                "step_in_epoch": int(step),
                "global_step": int(global_step),
                "loss": float(loss.detach().item()),
                "loss_char": float(L_char.detach().item()),
                "loss_ssim": float(L_ssim.detach().item()),
                "loss_mse": float(L_mse.detach().item()),
                "loss_mae": float(L_mae.detach().item()),
                "loss_psnr": float(L_psnr.detach().item()),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "scaling_factor": float(getattr(raw_vae.config, "scaling_factor", 0.0)),
                "train_psnr_1ch": float(psnr_1),
                "train_ssim_1ch": float(ssim_1),
                "train_mse_1ch": float(mse_1),
                "train_mae_1ch": float(mae_1),
            }

            progress_bar.set_postfix(loss=round(logs["loss"], 6), lr=round(logs["lr"], 8))

            if accelerator.is_main_process:
                append_jsonl(train_metric_file, logs)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.unwrap_model(vae).save_pretrained(args.output_dir)
        logger.info(f"Final VAE saved to {args.output_dir}")

        has_val = (
            args.validation_h5_path
            or args.validation_h5_path_gf2
            or args.validation_h5_path_qb
            or args.validation_h5_path_wv3
        )
        if has_val:
            log_validation_h5(vae, args, accelerator, weight_dtype, step=global_step, tag_prefix="final_val_h5")

    accelerator.end_training()


if __name__ == "__main__":
    args = load_config()
    main(args)
