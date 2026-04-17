import inspect
import argparse
import ast
import contextlib
import gc
import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import yaml
import h5py

from utils.metrics import Q4_numpy, Q8_numpy, SAM_numpy, ERGAS_numpy, SCC_numpy
from core.components.salad_pan import DualBranchXSAdapter, UNetDualBranchXSModel
from core.pipelines.salad_pan import StableDiffusionDualBranchXSPipeline

check_min_version("0.36.0.dev0")
logger = get_logger(__name__)

_SSIM_KERNEL_CACHE: Dict[tuple, torch.Tensor] = {}
_LEGACY_CN = "control" + "net"
_LEGACY_CNX = _LEGACY_CN + "_xs"
_ADAPTER_WEIGHTS_FILE = "dual_branch_xs_adapter.pt"


def _legacy_key(suffix: str) -> str:
    return f"{_LEGACY_CN}_{suffix}"


def _legacy_xs_key(suffix: str) -> str:
    return f"{_LEGACY_CNX}_{suffix}"


# ---------------------------
# Local logging helpers
# ---------------------------
def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_image_safe(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def sanitize_name(name: str) -> str:
    name = str(name)
    for a, b in [(" ", "_"), ("/", "_"), ("\\", "_"), (":", "_")]:
        name = name.replace(a, b)
    return name


def save_validation_image_group(
    base_dir: Path,
    dataset_name: str,
    step: int,
    sample_order: int,
    dataset_idx: int,
    lrms_img,
    pan_img,
    gt_img,
    gen_img,
):
    sample_dir = (
        base_dir
        / sanitize_name(dataset_name)
        / f"step_{int(step):08d}"
        / f"{int(sample_order):04d}_idx{int(dataset_idx)}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image_safe(lrms_img, sample_dir / "lrms.png")
    save_image_safe(pan_img, sample_dir / "pan.png")
    save_image_safe(gt_img, sample_dir / "gt.png")
    save_image_safe(gen_img, sample_dir / "gen.png")


# ---------------------------
# Config helpers
# ---------------------------
def _literal(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        if "," in x:
            return [s.strip() for s in x.split(",") if s.strip()]
        return [x]
    return [x]


def _require_keys(cfg: dict, required_keys: List[str]):
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(
            "Missing required config keys:\n" + "\n".join([f"  - {k}" for k in missing])
        )


def load_config():
    ap = argparse.ArgumentParser(
        description=(
            "Strict-config dual-branch XS spa/spe with 1ch VAE "
            "(per-band generation + multi-band SAM/ERGAS loss)."
        ),
        add_help=True,
    )
    ap.add_argument("--config", type=str, required=True, help="YAML config file path")
    ap.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help='Override fields in YAML, e.g.: -o train_batch_size=2 -o "validation_indices=[0,1]"',
    )
    cli, unknown = ap.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unused CLI args: {unknown}")

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cli.config} did not parse into a dict.")

    for item in cli.override:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item} (should be key=value)")
        k, v = item.split("=", 1)
        cfg[k.strip()] = _literal(v.strip())

    legacy_to_new = {
        "unet_xs_model_name_or_path": "unet_adapter_model_name_or_path",
        "adapter_xs_size_ratio": "adapter_size_ratio",
        "adapter_xs_learn_time_embedding": "adapter_learn_time_embedding",
        "adapter_xs_time_embedding_mix": "adapter_time_embedding_mix",
        _legacy_xs_key("size_ratio"): "adapter_size_ratio",
        _legacy_xs_key("learn_time_embedding"): "adapter_learn_time_embedding",
        _legacy_xs_key("time_embedding_mix"): "adapter_time_embedding_mix",
        _legacy_key("conditioning_scale_spa"): "conditioning_scale_spa",
        _legacy_key("conditioning_scale_spe"): "conditioning_scale_spe",
    }
    for legacy_key, new_key in legacy_to_new.items():
        if new_key not in cfg and legacy_key in cfg:
            cfg[new_key] = cfg[legacy_key]

    train_h5_paths = _as_list(cfg.get("train_h5_paths", None))
    if train_h5_paths is None:
        raise KeyError("Missing required config `train_h5_paths`.")
    cfg["train_h5_paths"] = train_h5_paths

    validation_h5_paths = _as_list(cfg.get("validation_h5_paths", None))
    cfg["validation_h5_paths"] = validation_h5_paths

    required_keys = [
        # base
        "pretrained_model_name_or_path",
        "vae_path",
        "output_dir",
        "logging_dir",
        "local_files_only",
        "revision",
        "variant",
        "tokenizer_name",
        "seed",
        # data
        "train_h5_paths",
        "train_h5_names",
        "validation_h5_paths",
        "validation_h5_names",
        "h5_keys",
        "resolution",
        "range_clip_min",
        "range_clip_max",
        "range_clip_max_map",
        "discard_out_of_range",
        "max_train_samples",
        # prompts
        "dataset_prompts",
        "band_prompts",
        "proportion_empty_prompts",
        "use_prompts_in_validation",
        # train
        "train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_train_steps",
        "dataloader_num_workers",
        "mixed_precision",
        "allow_tf32",
        "scale_lr",
        "gradient_checkpointing",
        "enable_xformers_memory_efficient_attention",
        # optimizer / scheduler
        "learning_rate",
        "use_8bit_adam",
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
        # xs / adapter
        "unet_adapter_model_name_or_path",
        "adapter_size_ratio",
        "adapter_learn_time_embedding",
        "adapter_time_embedding_mix",
        "conditioning_scale_spa",
        "conditioning_scale_spe",
        # losses
        "lambda_x0",
        "lambda_ssim",
        "lambda_psnr",
        "lambda_sam",
        "lambda_ergas",
        "ergas_ratio",
        "sam_eps",
        "ergas_eps",
        # validation
        "validation_count",
        "validation_indices",
        "val_num_inference_steps",
        "validation_steps",
        "val_band_batch_size",
        "save_validation_rgb",
        # checkpoint / resume
        "checkpointing_steps",
        "checkpoints_total_limit",
        "checkpoint_mode",
        "resume_from_checkpoint",
        # sampling / batching
        "enable_long_term_equal_sampling",
        "steps_per_epoch",
        "equal_sampling_strategy",
        "require_full_bands_in_batch",
        "same_noise_for_all_bands",
        "vae_latent_mode",
    ]
    _require_keys(cfg, required_keys)

    if int(cfg["resolution"]) % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8.")

    if not isinstance(cfg["h5_keys"], dict):
        raise ValueError("`h5_keys` must be a dict.")
    for k in ["gt", "lms", "pan"]:
        if k not in cfg["h5_keys"]:
            raise KeyError(f"`h5_keys` must contain `{k}`.")

    if cfg["train_h5_names"] is None:
        raise ValueError("`train_h5_names` must be explicitly provided.")
    if len(list(cfg["train_h5_names"])) != len(cfg["train_h5_paths"]):
        raise ValueError("train_h5_names length must match train_h5_paths.")

    if cfg["validation_h5_paths"] is not None:
        if cfg["validation_h5_names"] is None:
            raise ValueError("`validation_h5_names` must be explicitly provided when validation_h5_paths is set.")
        if len(list(cfg["validation_h5_names"])) != len(cfg["validation_h5_paths"]):
            raise ValueError("validation_h5_names length must match validation_h5_paths.")
    else:
        if cfg["validation_h5_names"] is not None and len(list(cfg["validation_h5_names"])) > 0:
            raise ValueError("validation_h5_names should be null/empty when validation_h5_paths is null.")

    if str(cfg["vae_latent_mode"]).strip().lower() not in ("sample", "mode"):
        raise ValueError("`vae_latent_mode` must be 'sample' or 'mode'.")

    if str(cfg["checkpoint_mode"]).strip().lower() not in ("light", "full"):
        raise ValueError("`checkpoint_mode` must be 'light' or 'full'.")

    if str(cfg["equal_sampling_strategy"]).strip().lower() not in ("round_robin", "random"):
        raise ValueError("`equal_sampling_strategy` must be 'round_robin' or 'random'.")

    if cfg["local_files_only"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

    for p in cfg["train_h5_paths"]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Train H5 not found: {p}")

    if cfg["validation_h5_paths"] is not None:
        for p in cfg["validation_h5_paths"]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Validation H5 not found: {p}")

    return SimpleNamespace(**cfg)


def _resolve_from_map(mapping: Any, ds_name: Optional[str], h5_path: str, default: Any = None):
    if mapping is None:
        return default
    if isinstance(mapping, (int, float, str)):
        return mapping
    if not isinstance(mapping, dict) or len(mapping) == 0:
        return default

    name = "" if ds_name is None else str(ds_name)
    stem = Path(h5_path).stem
    full = str(h5_path)

    if name in mapping:
        return mapping[name]
    if stem in mapping:
        return mapping[stem]

    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    if stem.lower() in lower_map:
        return lower_map[stem.lower()]

    name_l, stem_l, full_l = name.lower(), stem.lower(), full.lower()
    for k_l, v in lower_map.items():
        if k_l and (k_l in name_l or k_l in stem_l or k_l in full_l):
            return v

    return default


def resolve_clip_max(args, ds_name: str, h5_path: str) -> float:
    v = _resolve_from_map(args.range_clip_max_map, ds_name, h5_path, default=args.range_clip_max)
    return float(v)


def resolve_dataset_prompt(args, ds_name: str, h5_path: str) -> str:
    v = _resolve_from_map(args.dataset_prompts, ds_name, h5_path, default="")
    return "" if v is None else str(v)


def resolve_band_prompts(args, ds_name: str, h5_path: str) -> Optional[List[str]]:
    v = _resolve_from_map(args.band_prompts, ds_name, h5_path, default=None)
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    raise ValueError(f"Invalid band_prompts for dataset={ds_name}, file={h5_path}")


def build_prompt_text(dataset_prompt: str, band_prompt: Optional[str]) -> str:
    dataset_prompt = (dataset_prompt or "").strip()
    if band_prompt is None:
        return dataset_prompt
    band_prompt = str(band_prompt).strip()
    if not dataset_prompt:
        return band_prompt
    return f"{dataset_prompt} {band_prompt}"


# ---------------------------
# Dataset
# ---------------------------
class H5PanSharpenMultiBandImageDataset(torch.utils.data.Dataset):
    """
    One sample = one image (all bands)
    """

    def __init__(
        self,
        h5_path: str,
        keys: dict,
        resolution: int,
        clip_min: float,
        clip_max: float,
        discard_out_of_range: bool,
        tokenizer,
        proportion_empty_prompts: float,
        max_train_samples,
        seed,
        null_input_ids: torch.Tensor,
        dataset_name: str,
        dataset_prompt: str,
        band_prompts: Optional[List[str]],
    ):
        super().__init__()
        self.h5_path = h5_path
        self.keys = keys
        self.resolution = int(resolution)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.discard_out = bool(discard_out_of_range)
        self.tokenizer = tokenizer
        self.p_empty = float(proportion_empty_prompts)

        self.dataset_name = str(dataset_name)
        self.dataset_prompt = str(dataset_prompt)
        self.band_prompts = band_prompts[:] if band_prompts is not None else None

        self.null_input_ids = null_input_ids.detach().cpu().long()

        self.dataset_prompt_ids = self.null_input_ids
        self.prompt_ids_by_band: Optional[List[torch.Tensor]] = None

        if self.dataset_prompt.strip():
            _tok = tokenizer([self.dataset_prompt], padding="max_length", truncation=True, return_tensors="pt")
            self.dataset_prompt_ids = _tok.input_ids[0].detach().cpu().long()
        else:
            self.dataset_prompt_ids = self.null_input_ids

        if self.band_prompts is not None and len(self.band_prompts) > 0:
            combined_texts = [build_prompt_text(self.dataset_prompt, bp) for bp in self.band_prompts]
            _tok = tokenizer(combined_texts, padding="max_length", truncation=True, return_tensors="pt")
            self.prompt_ids_by_band = [t.detach().cpu().long() for t in _tok.input_ids]

        with h5py.File(self.h5_path, "r") as f:
            gt = f[self.keys["gt"]]
            lms = f[self.keys["lms"]]
            pan = f[self.keys["pan"]]
            N = int(gt.shape[0])
            C = int(gt.shape[1])

            assert lms.shape[0] == N and lms.shape[1] == C
            assert pan.shape[0] == N and pan.shape[1] == 1
            assert gt.shape[2:] == lms.shape[2:] == pan.shape[2:], "H,W mismatch"

            if self.discard_out:
                keep_n = []
                out_cnt = 0
                for i in range(N):
                    mn = min(gt[i].min(), lms[i].min(), pan[i].min())
                    mx = max(gt[i].max(), lms[i].max(), pan[i].max())
                    if (mn < self.clip_min) or (mx > self.clip_max):
                        out_cnt += 1
                    else:
                        keep_n.append(i)
                logger.info(
                    f"[H5Dataset-MS] file={Path(self.h5_path).name} total_imgs={N}, kept_imgs={len(keep_n)}, "
                    f"discarded_imgs={out_cnt} (outside [{self.clip_min},{self.clip_max}])"
                )
                base_indices = keep_n
            else:
                base_indices = list(range(N))

        if seed is not None:
            random.Random(seed).shuffle(base_indices)
        if max_train_samples is not None:
            base_indices = base_indices[: int(max_train_samples)]

        self.indices = [int(i) for i in base_indices]
        self._h5 = None
        self._gt = None
        self._lms = None
        self._pan = None

    def _open_if_needed(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._gt = self._h5[self.keys["gt"]]
            self._lms = self._h5[self.keys["lms"]]
            self._pan = self._h5[self.keys["pan"]]

    def __len__(self):
        return len(self.indices)

    def _resize_if_needed(self, x: torch.Tensor, size_hw):
        if tuple(x.shape[-2:]) != tuple(size_hw):
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return x

    def _pick_input_ids_by_band(self, C: int) -> torch.Tensor:
        if self.prompt_ids_by_band is None:
            if self.p_empty > 0 and random.random() < self.p_empty:
                return self.null_input_ids.clone()
            return self.dataset_prompt_ids.clone()

        ids = []
        for c in range(C):
            if self.p_empty > 0 and random.random() < self.p_empty:
                ids.append(self.null_input_ids)
            elif 0 <= c < len(self.prompt_ids_by_band):
                ids.append(self.prompt_ids_by_band[c])
            else:
                ids.append(self.dataset_prompt_ids)
        return torch.stack([t.clone() for t in ids], dim=0)

    def __getitem__(self, idx):
        self._open_if_needed()
        img_idx = self.indices[idx]

        gtC = np.array(self._gt[img_idx], dtype=np.float32)
        lmsC = np.array(self._lms[img_idx], dtype=np.float32)
        pan = np.array(self._pan[img_idx], dtype=np.float32)

        gtC = np.clip(gtC, self.clip_min, self.clip_max)
        lmsC = np.clip(lmsC, self.clip_min, self.clip_max)
        pan = np.clip(pan, self.clip_min, self.clip_max)

        scale = 1.0 / self.clip_max
        gtC = gtC * scale
        lmsC = lmsC * scale
        pan = pan * scale

        gt_ms = gtC * 2.0 - 1.0

        H, W = gt_ms.shape[-2:]
        if (H != self.resolution) or (W != self.resolution):
            gt_t = torch.from_numpy(gt_ms[None])
            lms_t = torch.from_numpy(lmsC[None])
            pan_t = torch.from_numpy(pan[None])
            gt_t = self._resize_if_needed(gt_t, (self.resolution, self.resolution))
            lms_t = self._resize_if_needed(lms_t, (self.resolution, self.resolution))
            pan_t = self._resize_if_needed(pan_t, (self.resolution, self.resolution))
            gt_ms = gt_t[0].numpy()
            lmsC = lms_t[0].numpy()
            pan = pan_t[0].numpy()

        C = int(gt_ms.shape[0])
        input_ids = self._pick_input_ids_by_band(C)

        return {
            "gt_ms": torch.from_numpy(gt_ms).float(),
            "lms_ms": torch.from_numpy(lmsC).float(),
            "pan": torch.from_numpy(pan).float(),
            "input_ids": input_ids.long(),
            "num_bands": torch.tensor(C, dtype=torch.int64),
        }


def collate_fn(examples):
    gt_ms = torch.stack([e["gt_ms"] for e in examples]).contiguous().float()
    lms_ms = torch.stack([e["lms_ms"] for e in examples]).contiguous().float()
    pan = torch.stack([e["pan"] for e in examples]).contiguous().float()

    ids_list = []
    for e in examples:
        ids = e["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids_list.append(ids)

    max_C = max(x.shape[0] for x in ids_list)
    padded = []
    for x in ids_list:
        if x.shape[0] < max_C:
            pad = x[-1:].repeat(max_C - x.shape[0], 1)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    input_ids = torch.stack(padded).contiguous().long()
    num_bands = torch.stack([e["num_bands"] for e in examples]).contiguous().long()

    return {
        "gt_ms": gt_ms,
        "lms_ms": lms_ms,
        "pan": pan,
        "input_ids": input_ids,
        "num_bands": num_bands,
    }


# ---------------------------
# Infinite sampler
# ---------------------------
class InfiniteDistributedRandomSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        seed: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        chunk_size: int,
    ):
        self.dataset = dataset
        self.n = len(dataset)
        if self.n <= 0:
            raise ValueError("InfiniteDistributedRandomSampler: dataset is empty")
        self.seed = int(seed)
        self.num_replicas = int(max(1, num_replicas))
        self.rank = int(rank)
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"Invalid rank={rank} for num_replicas={self.num_replicas}")
        self.shuffle = bool(shuffle)
        self.chunk_size = int(max(32, chunk_size))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000003)

        while True:
            if self.shuffle:
                idx = torch.randint(
                    low=0,
                    high=self.n,
                    size=(self.num_replicas * self.chunk_size,),
                    generator=g,
                    dtype=torch.int64,
                ).tolist()
            else:
                idx = list(range(self.n)) * self.num_replicas

            for j in range(self.rank, len(idx), self.num_replicas):
                yield idx[j]

    def __len__(self):
        return 2**31 - 1


# ---------------------------
# Metrics / losses
# ---------------------------
def calc_mse(x, y):
    return F.mse_loss(x, y, reduction="mean")


def calc_mae(x, y):
    return F.l1_loss(x, y, reduction="mean")


def calc_psnr(x, y, eps=1e-10):
    mse = calc_mse(x, y).clamp(min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return g / g.sum()


def _create_ssim_kernel(channels: int, window_size: int, sigma: float, device, dtype):
    g1d = _gaussian_window_1d(window_size, sigma, device, dtype)
    g2d = torch.outer(g1d, g1d)
    return g2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def _get_ssim_kernel(channels: int, window_size: int, sigma: float, device, dtype):
    key = (str(device), str(dtype), int(channels), int(window_size), float(sigma))
    k = _SSIM_KERNEL_CACHE.get(key, None)
    if k is None:
        k = _create_ssim_kernel(channels, window_size, sigma, device, dtype)
        _SSIM_KERNEL_CACHE[key] = k
    return k


def calc_ssim(x, y, window_size=11, sigma=1.5):
    C1 = (0.01**2)
    C2 = (0.03**2)
    x = x.float()
    y = y.float()
    _, c, _, _ = x.shape
    kernel = _get_ssim_kernel(c, window_size, sigma, x.device, x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, groups=c, padding=padding)
    mu_y = F.conv2d(y, kernel, groups=c, padding=padding)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, groups=c, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, groups=c, padding=padding) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=c, padding=padding) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


def ssim_loss_gt(recon01, gt01):
    return 1.0 - calc_ssim(recon01, gt01)


def sam_torch(pred: torch.Tensor, gt: torch.Tensor, eps: float) -> torch.Tensor:
    pred = pred.float()
    gt = gt.float()
    dot = (pred * gt).sum(dim=1)
    n1 = torch.sqrt((pred * pred).sum(dim=1).clamp_min(eps))
    n2 = torch.sqrt((gt * gt).sum(dim=1).clamp_min(eps))
    cos = (dot / (n1 * n2).clamp_min(eps)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos)
    return ang.mean()


def ergas_torch(pred: torch.Tensor, gt: torch.Tensor, ratio: float, eps: float) -> torch.Tensor:
    pred = pred.float()
    gt = gt.float()
    diff2 = (pred - gt) ** 2
    rmse_c = torch.sqrt(diff2.mean(dim=(0, 2, 3)).clamp_min(eps))
    mean_c = gt.mean(dim=(0, 2, 3)).abs().clamp_min(eps)
    ergas = (100.0 / float(ratio)) * torch.sqrt(((rmse_c / mean_c) ** 2).mean())
    return ergas


# ---------------------------
# Text encoder loader
# ---------------------------
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, local_files_only: bool):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        local_files_only=local_files_only,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    if model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    raise ValueError(f"{model_class} is not supported.")


# ---------------------------
# Validation
# ---------------------------
@torch.no_grad()
def log_validation_h5_xs_1ch_multi(
    vae,
    text_encoder,
    tokenizer,
    unet_xs,
    args,
    accelerator,
    weight_dtype,
    step,
    is_final_validation: bool = False,
):
    val_paths = args.validation_h5_paths
    if not val_paths:
        return
    if not accelerator.is_main_process:
        return

    logger.info("Running validation (local save mode)...")

    base_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    scheduler = UniPCMultistepScheduler.from_config(base_scheduler.config)

    unet_xs_eval = accelerator.unwrap_model(unet_xs)
    unet_xs_eval = unet_xs_eval._orig_mod if is_compiled_module(unet_xs_eval) else unet_xs_eval
    unet_xs_eval.eval()

    pipe = StableDiffusionDualBranchXSPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_xs_eval,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        adapter=None,
        **{_LEGACY_CN: None},
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(accelerator.device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    clip_min = float(args.range_clip_min)
    res = int(args.resolution)
    val_count = int(args.validation_count)
    specified_idx = args.validation_indices
    drop_oob = bool(args.discard_out_of_range)
    band_bs = int(args.val_band_batch_size)

    conditioning_scale_spa = float(args.conditioning_scale_spa)
    conditioning_scale_spe = float(args.conditioning_scale_spe)
    use_prompts_in_val = bool(args.use_prompts_in_validation)

    autocast_ctx = contextlib.nullcontext()
    if (not is_final_validation) and accelerator.device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=weight_dtype)

    metrics_file = Path(args.output_dir) / "validation_metrics.jsonl"
    vis_dir = Path(args.output_dir) / "validation_vis"

    def _resize_np(x, size_hw):
        t = torch.from_numpy(x[None])
        t = F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)
        return t[0].numpy()

    val_names = list(args.validation_h5_names)

    for vpath, vname_raw in zip(val_paths, val_names):
        clip_max = resolve_clip_max(args, vname_raw, vpath)
        ds_prompt = resolve_dataset_prompt(args, vname_raw, vpath) if use_prompts_in_val else ""
        band_prompts = resolve_band_prompts(args, vname_raw, vpath) if use_prompts_in_val else None
        vname = sanitize_name(vname_raw)

        all_mse, all_mae, all_psnr, all_ssim = [], [], [], []
        all_q, all_sam, all_ergas, all_scc = [], [], [], []

        with h5py.File(vpath, "r") as f:
            gt_ds = f[args.h5_keys["gt"]]
            lms_ds = f[args.h5_keys["lms"]]
            pan_ds = f[args.h5_keys["pan"]]
            N = int(gt_ds.shape[0])
            C_global = int(gt_ds.shape[1])

            if isinstance(specified_idx, dict):
                idxs = specified_idx.get(vname_raw, None)
                if idxs is None:
                    idxs = specified_idx.get(vname, None)
                if idxs is None:
                    rng = np.random.default_rng(args.seed)
                    idxs = rng.choice(N, size=min(val_count, N), replace=False).tolist()
                else:
                    idxs = list(idxs)
            elif specified_idx is None:
                rng = np.random.default_rng(args.seed)
                idxs = rng.choice(N, size=min(val_count, N), replace=False).tolist()
            else:
                idxs = list(specified_idx)

            idxs = [int(i) for i in idxs if 0 <= int(i) < N]
            if len(idxs) == 0:
                rng = np.random.default_rng(args.seed)
                idxs = rng.choice(N, size=min(val_count, N), replace=False).tolist()

            if val_count > 0:
                idxs = idxs[: min(val_count, len(idxs))]

            for i, idx in enumerate(idxs):
                gtC = np.array(gt_ds[idx], dtype=np.float32)
                lmsC = np.array(lms_ds[idx], dtype=np.float32)
                pan = np.array(pan_ds[idx], dtype=np.float32)

                if drop_oob:
                    mn = min(gtC.min(), lmsC.min(), pan.min())
                    mx = max(gtC.max(), lmsC.max(), pan.max())
                    if (mn < clip_min) or (mx > clip_max):
                        logger.info(
                            f"[Val:{vname}] skip idx={idx} out-of-range [{mn:.1f},{mx:.1f}] not in [{clip_min},{clip_max}]"
                        )
                        continue

                gtC = np.clip(gtC, clip_min, clip_max) / clip_max
                lmsC = np.clip(lmsC, clip_min, clip_max) / clip_max
                pan = np.clip(pan, clip_min, clip_max) / clip_max

                H, W = gtC.shape[-2:]
                if (H != res) or (W != res):
                    gtC = _resize_np(gtC, (res, res))
                    lmsC = _resize_np(lmsC, (res, res))
                    pan = _resize_np(pan, (res, res))

                C = int(gtC.shape[0])
                gen_bands = []

                for start in range(0, C, band_bs):
                    end = min(C, start + band_bs)
                    cond_list = []
                    prompts = []

                    for c in range(start, end):
                        lms_band = lmsC[c : c + 1]
                        lms_rep4 = np.repeat(lms_band, 4, 0)
                        cond5 = np.concatenate([lms_rep4, pan], axis=0)
                        cond_list.append(cond5)

                        if use_prompts_in_val:
                            bp = None
                            if band_prompts is not None and 0 <= c < len(band_prompts):
                                bp = band_prompts[c]
                            prompts.append(build_prompt_text(ds_prompt, bp))
                        else:
                            prompts.append("")

                    cond_arr = np.stack(cond_list, axis=0)
                    cond_t = torch.from_numpy(cond_arr).to(device=accelerator.device, dtype=weight_dtype)

                    with autocast_ctx:
                        out = pipe(
                            prompt=prompts,
                            image=cond_t,
                            num_inference_steps=int(args.val_num_inference_steps),
                            guidance_scale=1.0,
                            generator=generator,
                            output_type="pt",
                            conditioning_scale=1.0,
                            conditioning_scale_spa=conditioning_scale_spa,
                            conditioning_scale_spe=conditioning_scale_spe,
                        )
                        gen_band_01 = out.images

                    if not torch.is_tensor(gen_band_01):
                        gen_band_01 = torch.stack(gen_band_01, dim=0)
                    gen_band_01 = gen_band_01.to(accelerator.device, dtype=torch.float32)
                    gen_bands.append(gen_band_01)

                gen_band_01 = torch.cat(gen_bands, dim=0)
                gen_ms_01 = gen_band_01[:, 0, :, :].unsqueeze(0)
                gt_ms_01 = torch.from_numpy(gtC).unsqueeze(0).to(accelerator.device, dtype=torch.float32)

                mse_i = calc_mse(gen_ms_01, gt_ms_01).item()
                mae_i = calc_mae(gen_ms_01, gt_ms_01).item()
                psnr_i = calc_psnr(gen_ms_01, gt_ms_01).item()
                ssim_i = calc_ssim(gen_ms_01, gt_ms_01).item()

                all_mse.append(mse_i)
                all_mae.append(mae_i)
                all_psnr.append(psnr_i)
                all_ssim.append(ssim_i)

                gen_np = gen_ms_01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                gt_np = gt_ms_01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

                q_i = float("nan")
                try:
                    if C == 4:
                        q_i = float(Q4_numpy(gt_np, gen_np))
                    elif C == 8:
                        q_i = float(Q8_numpy(gt_np, gen_np))
                except Exception as e:
                    logger.warning(f"Q-metric failed {vname} idx={idx}: {e}")

                try:
                    sam_i = float(SAM_numpy(gt_np, gen_np))
                except Exception as e:
                    logger.warning(f"SAM failed {vname} idx={idx}: {e}")
                    sam_i = float("nan")
                try:
                    ergas_i = float(ERGAS_numpy(gt_np, gen_np))
                except Exception as e:
                    logger.warning(f"ERGAS failed {vname} idx={idx}: {e}")
                    ergas_i = float("nan")
                try:
                    scc_i = float(SCC_numpy(gt_np, gen_np))
                except Exception as e:
                    logger.warning(f"SCC failed {vname} idx={idx}: {e}")
                    scc_i = float("nan")

                all_q.append(q_i)
                all_sam.append(sam_i)
                all_ergas.append(ergas_i)
                all_scc.append(scc_i)

                append_jsonl(
                    metrics_file,
                    {
                        "time": datetime.now().isoformat(),
                        "step": int(step),
                        "dataset": vname,
                        "sample_kind": "fixed",
                        "sample_order": int(i),
                        "dataset_idx": int(idx),
                        "bands": int(C),
                        "mse": float(mse_i),
                        "mae": float(mae_i),
                        "psnr": float(psnr_i),
                        "ssim": float(ssim_i),
                        "q": float(q_i),
                        "sam": float(sam_i),
                        "ergas": float(ergas_i),
                        "scc": float(scc_i),
                        "clip_max": float(clip_max),
                        "use_prompts": bool(use_prompts_in_val),
                    },
                )

                if bool(args.save_validation_rgb):
                    rgb_ch = min(3, C)
                    gen_rgb_pil = to_pil_image(gen_ms_01[0, :rgb_ch].clamp(0, 1).cpu())
                    gt_rgb_pil = to_pil_image(gt_ms_01[0, :rgb_ch].clamp(0, 1).cpu())
                    lrms_rgb_pil = to_pil_image(torch.from_numpy(lmsC[:rgb_ch]).clamp(0, 1))
                    pan_vis = np.repeat(pan, 3, axis=0)
                    pan_pil = to_pil_image(torch.from_numpy(pan_vis).clamp(0, 1))

                    save_validation_image_group(
                        vis_dir,
                        vname,
                        step,
                        i,
                        idx,
                        lrms_rgb_pil,
                        pan_pil,
                        gt_rgb_pil,
                        gen_rgb_pil,
                    )

            if len(all_mse) > 0:
                append_jsonl(
                    metrics_file,
                    {
                        "time": datetime.now().isoformat(),
                        "step": int(step),
                        "dataset": vname,
                        "sample_kind": "mean",
                        "bands": int(C_global),
                        "num_samples": int(len(all_mse)),
                        "mean_mse": float(np.nanmean(all_mse)),
                        "mean_mae": float(np.nanmean(all_mae)),
                        "mean_psnr": float(np.nanmean(all_psnr)),
                        "mean_ssim": float(np.nanmean(all_ssim)),
                        "mean_q": float(np.nanmean(all_q)),
                        "mean_sam": float(np.nanmean(all_sam)),
                        "mean_ergas": float(np.nanmean(all_ergas)),
                        "mean_scc": float(np.nanmean(all_scc)),
                        "clip_max": float(clip_max),
                        "use_prompts": bool(use_prompts_in_val),
                    },
                )

                logger.info(
                    f"[Validation:{vname}] step={step} mean_psnr={np.nanmean(all_psnr):.4f} "
                    f"mean_ssim={np.nanmean(all_ssim):.4f} mean_q={np.nanmean(all_q):.4f} "
                    f"mean_sam={np.nanmean(all_sam):.4f} mean_ergas={np.nanmean(all_ergas):.4f} "
                    f"mean_scc={np.nanmean(all_scc):.4f}"
                )

    unet_xs_eval.train()
    del pipe
    gc.collect()
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------
# Utils
# ---------------------------
def unwrap_model(accelerator, model):
    m = accelerator.unwrap_model(model)
    return m._orig_mod if is_compiled_module(m) else m


def get_trainable_state_dict(accelerator, model):
    base = unwrap_model(accelerator, model)
    trainable_names = {n for n, p in base.named_parameters() if p.requires_grad}
    full = base.state_dict()
    return {k: v.detach().cpu() for k, v in full.items() if k in trainable_names}


def _save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_checkpoints(output_dir: str):
    if not os.path.isdir(output_dir):
        return []
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    items = []
    for d in dirs:
        try:
            step = int(d.split("-")[1])
            items.append((step, d))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    return items


def _resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str):
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        items = _list_checkpoints(output_dir)
        return os.path.join(output_dir, items[-1][1]) if items else None
    if os.path.isdir(resume_from_checkpoint):
        return resume_from_checkpoint
    cand = os.path.join(output_dir, resume_from_checkpoint)
    if os.path.isdir(cand):
        return cand
    return None


def _save_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_rng_state(state: dict):
    try:
        if state.get("python", None) is not None:
            random.setstate(state["python"])
        if state.get("numpy", None) is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch", None) is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda", None) is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        logger.warning(f"Failed to restore RNG state: {e}")


def _torch_load_compat(path: str, *, map_location="cpu", weights_only: Optional[bool] = None):
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": map_location}
    if weights_only is not None and "weights_only" in sig.parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def save_checkpoint_light(
    accelerator: Accelerator,
    unet_xs,
    optimizer,
    lr_scheduler,
    save_dir: str,
    *,
    global_step: int,
    epoch: int,
    resume_step: int,
    args,
):
    os.makedirs(save_dir, exist_ok=True)

    adapter_state = get_trainable_state_dict(accelerator, unet_xs)
    torch.save(adapter_state, os.path.join(save_dir, _ADAPTER_WEIGHTS_FILE))

    accelerator.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    accelerator.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    torch.save(_save_rng_state(), os.path.join(save_dir, "rng_state.pt"))

    training_state = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "resume_step": int(resume_step),
        "max_train_steps": int(args.max_train_steps),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "enable_long_term_equal_sampling": bool(args.enable_long_term_equal_sampling),
        "steps_per_epoch": int(args.steps_per_epoch) if args.steps_per_epoch is not None else None,
        "equal_sampling_strategy": str(args.equal_sampling_strategy),
        "same_noise_for_all_bands": bool(args.same_noise_for_all_bands),
        "vae_latent_mode": str(args.vae_latent_mode),
    }
    _save_json(training_state, os.path.join(save_dir, "training_state.json"))


def load_checkpoint_light(
    accelerator: Accelerator,
    unet_xs,
    optimizer,
    lr_scheduler,
    ckpt_dir: str,
):
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing training_state.json in {ckpt_dir}")
    training_state = _load_json(state_path)

    adapter_path = os.path.join(ckpt_dir, _ADAPTER_WEIGHTS_FILE)
    if not os.path.isfile(adapter_path):
        legacy_name = _LEGACY_CNX + "_adapter.pt"
        legacy_path = os.path.join(ckpt_dir, legacy_name)
        if os.path.isfile(legacy_path):
            adapter_path = legacy_path
        else:
            raise FileNotFoundError(
                f"Missing adapter weights in {ckpt_dir}: tried '{_ADAPTER_WEIGHTS_FILE}' and '{legacy_name}'"
            )
    adapter_state = _torch_load_compat(adapter_path, map_location="cpu", weights_only=True)

    base = unwrap_model(accelerator, unet_xs)
    missing, unexpected = base.load_state_dict(adapter_state, strict=False)
    if accelerator.is_main_process:
        if missing:
            logger.info(f"[Resume] adapter missing keys (ok): {len(missing)}")
        if unexpected:
            logger.info(f"[Resume] adapter unexpected keys (ok): {len(unexpected)}")

        trainable_names = {n for n, p in base.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing if k in trainable_names]
        unexpected_trainable = [k for k in unexpected if k in trainable_names]

        if len(missing_trainable) > 0:
            logger.warning(
                f"[Resume][WARN] Missing TRAINABLE keys: {len(missing_trainable)}. Examples: {missing_trainable[:20]}"
            )
        else:
            logger.info("[Resume] Sanity check passed: no missing trainable keys.")

        if len(unexpected_trainable) > 0:
            logger.warning(
                f"[Resume][WARN] Unexpected TRAINABLE keys: {len(unexpected_trainable)}. "
                f"Examples: {unexpected_trainable[:20]}"
            )

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    opt_state = _torch_load_compat(opt_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt_state)

    sch_path = os.path.join(ckpt_dir, "scheduler.pt")
    sch_state = _torch_load_compat(sch_path, map_location="cpu", weights_only=True)
    lr_scheduler.load_state_dict(sch_state)

    rng_path = os.path.join(ckpt_dir, "rng_state.pt")
    if os.path.isfile(rng_path):
        rng_state = _torch_load_compat(rng_path, map_location="cpu", weights_only=False)
        _load_rng_state(rng_state)

    accelerator.wait_for_everyone()
    return training_state


def _maybe_set_epoch(dataloader, epoch: int):
    s = getattr(dataloader, "sampler", None)
    if s is None and hasattr(dataloader, "dataloader"):
        s = getattr(dataloader.dataloader, "sampler", None)
    if hasattr(s, "set_epoch"):
        s.set_epoch(epoch)


# ---------------------------
# Main
# ---------------------------
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    def _as_float(x):
        return None if x is None else float(x)

    def _as_int(x):
        return None if x is None else int(x)

    def _as_bool(x):
        if isinstance(x, str):
            return x.lower() in ("1", "true", "yes", "y", "on")
        return bool(x)

    float_keys = [
        "range_clip_min",
        "range_clip_max",
        "learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_weight_decay",
        "adam_epsilon",
        "max_grad_norm",
        "lr_power",
        "adapter_size_ratio",
        "adapter_time_embedding_mix",
        "conditioning_scale_spa",
        "conditioning_scale_spe",
        "lambda_x0",
        "lambda_ssim",
        "lambda_psnr",
        "lambda_sam",
        "lambda_ergas",
        "ergas_ratio",
        "sam_eps",
        "ergas_eps",
        "proportion_empty_prompts",
    ]
    int_keys = [
        "seed",
        "resolution",
        "train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_train_steps",
        "dataloader_num_workers",
        "lr_warmup_steps",
        "lr_num_cycles",
        "validation_count",
        "val_num_inference_steps",
        "validation_steps",
        "val_band_batch_size",
        "checkpointing_steps",
        "checkpoints_total_limit",
        "steps_per_epoch",
        "max_train_samples",
    ]
    bool_keys = [
        "local_files_only",
        "discard_out_of_range",
        "use_prompts_in_validation",
        "allow_tf32",
        "scale_lr",
        "gradient_checkpointing",
        "enable_xformers_memory_efficient_attention",
        "use_8bit_adam",
        "set_grads_to_none",
        "adapter_learn_time_embedding",
        "save_validation_rgb",
        "enable_long_term_equal_sampling",
        "require_full_bands_in_batch",
        "same_noise_for_all_bands",
    ]

    for k in float_keys:
        setattr(args, k, _as_float(getattr(args, k)))
    for k in int_keys:
        setattr(args, k, _as_int(getattr(args, k)))
    for k in bool_keys:
        setattr(args, k, _as_bool(getattr(args, k)))

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        with (Path(args.output_dir) / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=False)

    # tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
            local_files_only=args.local_files_only,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            local_files_only=args.local_files_only,
        )

    # text encoder
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
        args.revision,
        args.local_files_only,
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )

    # noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )

    # 1ch VAE
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=args.local_files_only)

    # UNet + XS
    if args.unet_adapter_model_name_or_path:
        logger.info("Loading existing UNetDualBranchXSModel")
        unet_xs = UNetDualBranchXSModel.from_pretrained(
            args.unet_adapter_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            local_files_only=args.local_files_only,
        )
    else:
        logger.info("Initializing dual-branch XS adapter from base UNet")
        base_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
            local_files_only=args.local_files_only,
        )

        adapter_xs = DualBranchXSAdapter.from_unet(
            base_unet,
            size_ratio=args.adapter_size_ratio,
            learn_time_embedding=args.adapter_learn_time_embedding,
            time_embedding_mix=args.adapter_time_embedding_mix,
            conditioning_channels=5,
            conditioning_channel_order="rgb",
        )
        unet_xs = UNetDualBranchXSModel.from_unet(base_unet, **{_LEGACY_CN: adapter_xs})

        del base_unet
        del adapter_xs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unet_xs.to(dtype=torch.float32)

    # Freeze modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    if hasattr(unet_xs, "freeze_unet_params"):
        unwrap_model(accelerator, unet_xs).freeze_unet_params()
    else:
        logger.warning("unet_xs has no freeze_unet_params(); all params may become trainable")

    base_model = unwrap_model(accelerator, unet_xs)
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        logger.info(
            f"[ParamCount] UNetXS total={total_params:,} ({total_params/1e6:.2f}M), "
            f"trainable={trainable_params:,} ({trainable_params/1e6:.2f}M), "
            f"ratio={trainable_params/max(1,total_params):.6f}"
        )

    unet_xs.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            if version.parse(xformers.__version__) == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable; consider upgrading.")
            if hasattr(unet_xs, "enable_xformers_memory_efficient_attention"):
                unet_xs.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not installed or unavailable.")

    if args.gradient_checkpointing:
        if hasattr(unet_xs, "enable_gradient_checkpointing"):
            unet_xs.enable_gradient_checkpointing()

    if unwrap_model(accelerator, unet_xs).dtype != torch.float32:
        raise ValueError("Start UNetXS in float32 before mixed precision training.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # optimizer
    trainable_params_list = [p for p in unet_xs.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes required: pip install bitsandbytes")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        trainable_params_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # dataset / dataloader
    null_tok = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt")
    null_input_ids = null_tok.input_ids[0]

    train_paths = list(args.train_h5_paths)
    train_names = list(args.train_h5_names)

    datasets = []
    for p, n in zip(train_paths, train_names):
        ds_clip_max = resolve_clip_max(args, n, p)
        ds_prompt = resolve_dataset_prompt(args, n, p)
        ds_band_prompts = resolve_band_prompts(args, n, p)

        logger.info(f"[ClipMax] train dataset={n}, file={Path(p).name}, clip_max={ds_clip_max}")
        if ds_prompt.strip():
            logger.info(f"[Prompt] train dataset={n}, prompt='{ds_prompt[:120]}'")
        if ds_band_prompts is not None:
            logger.info(f"[BandPrompt] train dataset={n}, band_prompts={len(ds_band_prompts)}")

        ds = H5PanSharpenMultiBandImageDataset(
            h5_path=p,
            keys=args.h5_keys,
            resolution=args.resolution,
            clip_min=args.range_clip_min,
            clip_max=ds_clip_max,
            discard_out_of_range=args.discard_out_of_range,
            tokenizer=tokenizer,
            proportion_empty_prompts=args.proportion_empty_prompts,
            max_train_samples=args.max_train_samples,
            seed=args.seed,
            null_input_ids=null_input_ids,
            dataset_name=n,
            dataset_prompt=ds_prompt,
            band_prompts=ds_band_prompts,
        )
        logger.info(f"[TrainDataset] name={n}, file={Path(p).name}, images={len(ds)}")
        datasets.append(ds)

    num_workers = args.dataloader_num_workers
    use_equal = args.enable_long_term_equal_sampling
    equal_strategy = str(args.equal_sampling_strategy).lower().strip()

    train_dataloader = None
    train_dataloaders = None
    train_dataset_total_len = int(sum(len(d) for d in datasets))

    if not use_equal:
        if len(datasets) == 1:
            train_dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset(datasets)

        train_sampler = None
        shuffle = True
        if accelerator.num_processes > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                seed=int(args.seed),
                drop_last=False,
            )
            shuffle = False

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0),
        )

        unet_xs, optimizer, train_dataloader = accelerator.prepare(unet_xs, optimizer, train_dataloader)
        trainable_params_list = [p for p in unet_xs.parameters() if p.requires_grad]
    else:
        train_dataloaders = []
        for ds in datasets:
            sampler = InfiniteDistributedRandomSampler(
                ds,
                seed=int(args.seed),
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                chunk_size=4096,
            )
            dl = torch.utils.data.DataLoader(
                ds,
                sampler=sampler,
                shuffle=False,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=bool(num_workers > 0),
            )
            train_dataloaders.append(dl)

        unet_xs, optimizer = accelerator.prepare(unet_xs, optimizer)
        trainable_params_list = [p for p in unet_xs.parameters() if p.requires_grad]

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=accelerator.device, dtype=weight_dtype)

    # steps/epochs
    if not use_equal:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    else:
        global_batch = int(args.train_batch_size) * int(accelerator.num_processes) * int(args.gradient_accumulation_steps)
        spe = args.steps_per_epoch
        if spe is None:
            spe = int(math.ceil(train_dataset_total_len / max(1, global_batch)))
        spe = int(max(1, spe))
        args.steps_per_epoch = spe
        num_update_steps_per_epoch = spe

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.lr_warmup_steps),
        num_training_steps=int(args.max_train_steps * 1.10),
        num_cycles=int(args.lr_num_cycles),
        power=float(args.lr_power),
    )

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num images (total) = {train_dataset_total_len}")
    if not use_equal:
        logger.info(f"  Num batches each epoch (per-rank, micro-batches) = {len(train_dataloader)}")
    else:
        logger.info(
            f"  Long-term equal sampling = True, strategy={equal_strategy}, steps_per_epoch={args.steps_per_epoch}"
        )
        logger.info(f"  Micro-batches per epoch (per-rank) = {args.steps_per_epoch * args.gradient_accumulation_steps}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device (images) = {args.train_batch_size}")
    logger.info(f"  Total train batch size (images) = {total_batch_size}")
    logger.info("  NOTE: effective UNet/VAE forward batch = images * bands (after flatten)")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    resume_step = 0

    # resume
    if args.resume_from_checkpoint:
        ckpt = _resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
        if ckpt is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
        else:
            accelerator.print(f"Resuming from checkpoint {ckpt}")
            state_path = os.path.join(ckpt, "training_state.json")
            if os.path.isfile(state_path):
                training_state = load_checkpoint_light(
                    accelerator=accelerator,
                    unet_xs=unet_xs,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ckpt_dir=ckpt,
                )
                global_step = int(training_state.get("global_step", 0))
            else:
                accelerator.load_state(ckpt)
                try:
                    bn = os.path.basename(ckpt)
                    global_step = int(bn.split("-")[1]) if bn.startswith("checkpoint-") else 0
                except Exception:
                    global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            if not use_equal:
                resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
            else:
                resume_step = global_step % num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    device_type = accelerator.device.type

    train_metric_file = Path(args.output_dir) / "train_metrics.jsonl"

    def _vae_encode_latents(gt_flat_1ch: torch.Tensor) -> torch.Tensor:
        posterior = vae.encode(gt_flat_1ch.to(dtype=weight_dtype)).latent_dist
        if args.vae_latent_mode == "mode":
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * vae.config.scaling_factor
        return latents

    def _make_noise_like_latents(latents: torch.Tensor, B: int, C: int) -> torch.Tensor:
        if not args.same_noise_for_all_bands:
            return torch.randn_like(latents)
        noise_img = torch.randn((B,) + tuple(latents.shape[1:]), device=latents.device, dtype=latents.dtype)
        return noise_img.repeat_interleave(C, dim=0)

    def _prepare_flat_batch(batch: dict):
        gt_ms = batch["gt_ms"].to(accelerator.device, dtype=torch.float32)
        lms_ms = batch["lms_ms"].to(accelerator.device, dtype=torch.float32)
        pan = batch["pan"].to(accelerator.device, dtype=torch.float32)
        input_ids = batch["input_ids"].to(accelerator.device)
        num_bands = batch["num_bands"].to(accelerator.device)

        B, max_C, H, W = gt_ms.shape
        if args.require_full_bands_in_batch:
            if not torch.all(num_bands == num_bands[0]):
                raise ValueError(
                    f"Mixed num_bands in one batch: {num_bands.tolist()}. "
                    "Set require_full_bands_in_batch=false to allow mixed-band batches."
                )
            C = int(num_bands[0].item())
        else:
            C = int(num_bands.max().item())

        gt_ms = gt_ms[:, :C]
        lms_ms = lms_ms[:, :C]
        input_ids = input_ids[:, :C]

        gt_flat = gt_ms.reshape(B * C, 1, H, W)
        lms_flat = lms_ms.reshape(B * C, 1, H, W)
        pan_flat = pan.repeat_interleave(C, dim=0)
        cond5 = torch.cat([lms_flat.repeat(1, 4, 1, 1), pan_flat], dim=1)
        ids_flat = input_ids.reshape(B * C, -1)

        return gt_ms, gt_flat, lms_ms, pan, cond5, ids_flat, (B, C, H, W)

    ckpt_every = int(args.checkpointing_steps or 0)
    val_every = int(args.validation_steps or 0)

    def _maybe_run_checkpoint_and_validation(epoch, resume_step_or_step, current_global_step):
        do_ckpt = (ckpt_every > 0) and (current_global_step % ckpt_every == 0)
        do_val = (val_every > 0) and (current_global_step % val_every == 0)

        if not (do_ckpt or do_val):
            return

        accelerator.wait_for_everyone()

        if do_ckpt and accelerator.is_main_process:
            if args.checkpoints_total_limit is not None:
                checkpoints = _list_checkpoints(args.output_dir)
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing = checkpoints[:num_to_remove]
                    logger.info(
                        f"Removing {len(removing)} old checkpoints: {', '.join([d for _, d in removing])}"
                    )
                    for _, ck in removing:
                        shutil.rmtree(os.path.join(args.output_dir, ck), ignore_errors=True)

            save_path = os.path.join(args.output_dir, f"checkpoint-{current_global_step}")
            ckpt_mode = str(args.checkpoint_mode).lower()
            if ckpt_mode == "light":
                save_checkpoint_light(
                    accelerator=accelerator,
                    unet_xs=unet_xs,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    save_dir=save_path,
                    global_step=current_global_step,
                    epoch=epoch,
                    resume_step=resume_step_or_step,
                    args=args,
                )
                logger.info(f"Saved LIGHT checkpoint to {save_path}")
            else:
                accelerator.save_state(save_path)
                logger.info(f"Saved FULL state to {save_path}")
                adapter_state = get_trainable_state_dict(accelerator, unet_xs)
                torch.save(adapter_state, os.path.join(save_path, _ADAPTER_WEIGHTS_FILE))

        accelerator.wait_for_everyone()

        if do_val and accelerator.is_main_process:
            log_validation_h5_xs_1ch_multi(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet_xs=unet_xs,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=current_global_step,
                is_final_validation=False,
            )

        accelerator.wait_for_everyone()

    # ---------------------------
    # Training loop
    # ---------------------------
    if not use_equal:
        for epoch in range(first_epoch, args.num_train_epochs):
            _maybe_set_epoch(train_dataloader, epoch)

            for step, batch in enumerate(train_dataloader):
                if epoch == first_epoch and step < resume_step:
                    continue

                with accelerator.accumulate(unet_xs):
                    gt_ms, gt_flat, lms_ms, pan, adapter_conditioning_image, ids_flat, shape_info = _prepare_flat_batch(batch)
                    B, C, H, W = shape_info

                    latents = _vae_encode_latents(gt_flat)
                    noise = _make_noise_like_latents(latents, B=B, C=C)

                    t_img = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=latents.device).long()
                    timesteps = t_img.repeat_interleave(C, dim=0)

                    alpha_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1).clamp(min=1e-6)
                    sqrt_alpha_bar = alpha_bar.sqrt()
                    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
                    noisy_latents = (sqrt_alpha_bar * latents.float() + sqrt_one_minus_alpha_bar * noise.float()).to(
                        dtype=weight_dtype
                    )

                    encoder_hidden_states = text_encoder(ids_flat, return_dict=False)[0]

                    unet_kwargs = {
                        "adapter_cond": adapter_conditioning_image.to(dtype=weight_dtype),
                        "conditioning_scale_spa": args.conditioning_scale_spa,
                        "conditioning_scale_spe": args.conditioning_scale_spe,
                    }
                    model_pred = unet_xs(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=True,
                        **unet_kwargs,
                    ).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        eps_pred = model_pred
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        eps_pred = sqrt_one_minus_alpha_bar * noisy_latents + sqrt_alpha_bar * model_pred
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    x0_pred = (noisy_latents - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

                    loss_x0_lat = F.l1_loss(x0_pred.float(), latents.float())
                    L_x0 = args.lambda_x0 * loss_x0_lat.float()

                    z = (x0_pred / vae.config.scaling_factor).to(dtype=weight_dtype)
                    recon_img = vae.decode(z).sample
                    recon_01_flat = ((recon_img.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
                    gt_01_flat = ((gt_flat.to(dtype=torch.float32) + 1.0) * 0.5).clamp(0, 1)

                    recon_ms_01 = recon_01_flat.reshape(B, C, H, W)
                    gt_ms_01 = gt_01_flat.reshape(B, C, H, W)

                    autocast_off = (
                        torch.autocast("cuda", enabled=False) if device_type == "cuda" else contextlib.nullcontext()
                    )
                    with autocast_off:
                        if args.lambda_ssim > 0:
                            loss_ssim = ssim_loss_gt(recon_01_flat.float(), gt_01_flat.float())
                        else:
                            loss_ssim = torch.tensor(0.0, device=recon_01_flat.device)

                        if args.lambda_psnr > 0:
                            psnr_val = calc_psnr(recon_01_flat.float(), gt_01_flat.float())
                            loss_psnr = torch.clamp(1.0 - psnr_val / 50.0, min=0.0)
                        else:
                            loss_psnr = torch.tensor(0.0, device=recon_01_flat.device)

                        if args.lambda_sam > 0:
                            loss_sam = sam_torch(recon_ms_01.float(), gt_ms_01.float(), eps=args.sam_eps)
                        else:
                            loss_sam = torch.tensor(0.0, device=recon_01_flat.device)

                        if args.lambda_ergas > 0:
                            loss_ergas = ergas_torch(
                                recon_ms_01.float(), gt_ms_01.float(), ratio=args.ergas_ratio, eps=args.ergas_eps
                            )
                        else:
                            loss_ergas = torch.tensor(0.0, device=recon_01_flat.device)

                    loss = (
                        L_x0
                        + args.lambda_ssim * loss_ssim.float()
                        + args.lambda_psnr * loss_psnr.float()
                        + args.lambda_sam * loss_sam.float()
                        + args.lambda_ergas * loss_ergas.float()
                    ).float()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {
                        "time": datetime.now().isoformat(),
                        "epoch": int(epoch),
                        "step_in_epoch": int(step),
                        "global_step": int(global_step),
                        "loss": float(loss.detach().item()),
                        "loss_x0_lat": float(loss_x0_lat.detach().item()),
                        "loss_ssim": float(loss_ssim.detach().item()),
                        "loss_psnr": float(loss_psnr.detach().item()),
                        "loss_sam": float(loss_sam.detach().item()),
                        "loss_ergas": float(loss_ergas.detach().item()),
                        "lr": float(lr_scheduler.get_last_lr()[0]),
                        "bands_in_step": int(C),
                        "same_noise_for_all_bands": bool(args.same_noise_for_all_bands),
                        "vae_latent_mode": str(args.vae_latent_mode),
                    }
                    progress_bar.set_postfix(
                        loss=round(logs["loss"], 4),
                        lr=round(logs["lr"], 8),
                    )
                    if accelerator.is_main_process:
                        append_jsonl(train_metric_file, logs)

                    _maybe_run_checkpoint_and_validation(epoch, step, global_step)

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

    else:
        n_dl = len(train_dataloaders)
        if n_dl <= 0:
            raise ValueError("enable_long_term_equal_sampling=True but no train dataloaders built.")

        seed_base = int(args.seed)

        for epoch in range(first_epoch, args.num_train_epochs):
            for dl in train_dataloaders:
                _maybe_set_epoch(dl, epoch)
            iters = [iter(dl) for dl in train_dataloaders]

            def _choose_ds_id(step_idx: int) -> int:
                if equal_strategy == "random":
                    rr = random.Random(seed_base + int(step_idx))
                    return rr.randrange(n_dl)
                return int(step_idx % n_dl)

            if epoch == first_epoch and resume_step > 0:
                start_step_idx = int(global_step - resume_step)
                for s in range(int(resume_step)):
                    skip_step_idx = start_step_idx + s
                    ds_skip = _choose_ds_id(skip_step_idx)
                    for _ in range(int(args.gradient_accumulation_steps)):
                        _ = next(iters[ds_skip])

            step_in_epoch_start = int(resume_step) if epoch == first_epoch else 0

            for step_in_epoch in range(step_in_epoch_start, num_update_steps_per_epoch):
                ds_id = _choose_ds_id(int(global_step))

                for _micro in range(int(args.gradient_accumulation_steps)):
                    batch = next(iters[ds_id])

                    with accelerator.accumulate(unet_xs):
                        gt_ms, gt_flat, lms_ms, pan, adapter_conditioning_image, ids_flat, shape_info = _prepare_flat_batch(batch)
                        B, C, H, W = shape_info

                        latents = _vae_encode_latents(gt_flat)
                        noise = _make_noise_like_latents(latents, B=B, C=C)

                        t_img = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=latents.device).long()
                        timesteps = t_img.repeat_interleave(C, dim=0)

                        alpha_bar = alphas_cumprod[timesteps].view(-1, 1, 1, 1).clamp(min=1e-6)
                        sqrt_alpha_bar = alpha_bar.sqrt()
                        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
                        noisy_latents = (sqrt_alpha_bar * latents.float() + sqrt_one_minus_alpha_bar * noise.float()).to(
                            dtype=weight_dtype
                        )

                        encoder_hidden_states = text_encoder(ids_flat, return_dict=False)[0]

                        unet_kwargs = {
                            "adapter_cond": adapter_conditioning_image.to(dtype=weight_dtype),
                            "conditioning_scale_spa": args.conditioning_scale_spa,
                            "conditioning_scale_spe": args.conditioning_scale_spe,
                        }
                        model_pred = unet_xs(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=True,
                            **unet_kwargs,
                        ).sample

                        if noise_scheduler.config.prediction_type == "epsilon":
                            eps_pred = model_pred
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            eps_pred = sqrt_one_minus_alpha_bar * noisy_latents + sqrt_alpha_bar * model_pred
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        x0_pred = (noisy_latents - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
                        loss_x0_lat = F.l1_loss(x0_pred.float(), latents.float())
                        L_x0 = args.lambda_x0 * loss_x0_lat.float()

                        z = (x0_pred / vae.config.scaling_factor).to(dtype=weight_dtype)
                        recon_img = vae.decode(z).sample
                        recon_01_flat = ((recon_img.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
                        gt_01_flat = ((gt_flat.to(dtype=torch.float32) + 1.0) * 0.5).clamp(0, 1)

                        recon_ms_01 = recon_01_flat.reshape(B, C, H, W)
                        gt_ms_01 = gt_01_flat.reshape(B, C, H, W)

                        autocast_off = (
                            torch.autocast("cuda", enabled=False) if device_type == "cuda" else contextlib.nullcontext()
                        )
                        with autocast_off:
                            if args.lambda_ssim > 0:
                                loss_ssim = ssim_loss_gt(recon_ms_01.float(), gt_ms_01.float())
                            else:
                                loss_ssim = torch.tensor(0.0, device=recon_ms_01.device)

                            if args.lambda_psnr > 0:
                                psnr_val = calc_psnr(recon_ms_01.float(), gt_ms_01.float())
                                loss_psnr = torch.clamp(1.0 - psnr_val / 50.0, min=0.0)
                            else:
                                loss_psnr = torch.tensor(0.0, device=recon_ms_01.device)

                            if args.lambda_sam > 0:
                                loss_sam = sam_torch(recon_ms_01.float(), gt_ms_01.float(), eps=args.sam_eps)
                            else:
                                loss_sam = torch.tensor(0.0, device=recon_ms_01.device)

                            if args.lambda_ergas > 0:
                                loss_ergas = ergas_torch(
                                    recon_ms_01.float(), gt_ms_01.float(), ratio=args.ergas_ratio, eps=args.ergas_eps
                                )
                            else:
                                loss_ergas = torch.tensor(0.0, device=recon_ms_01.device)

                        loss = (
                            L_x0
                            + args.lambda_ssim * loss_ssim.float()
                            + args.lambda_psnr * loss_psnr.float()
                            + args.lambda_sam * loss_sam.float()
                            + args.lambda_ergas * loss_ergas.float()
                        ).float()

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        logs = {
                            "time": datetime.now().isoformat(),
                            "epoch": int(epoch),
                            "step_in_epoch": int(step_in_epoch),
                            "global_step": int(global_step),
                            "loss": float(loss.detach().item()),
                            "loss_x0_lat": float(loss_x0_lat.detach().item()),
                            "loss_ssim": float(loss_ssim.detach().item()),
                            "loss_psnr": float(loss_psnr.detach().item()),
                            "loss_sam": float(loss_sam.detach().item()),
                            "loss_ergas": float(loss_ergas.detach().item()),
                            "lr": float(lr_scheduler.get_last_lr()[0]),
                            "equal_sampling_ds_id": int(ds_id),
                            "bands_in_step": int(C),
                            "same_noise_for_all_bands": bool(args.same_noise_for_all_bands),
                            "vae_latent_mode": str(args.vae_latent_mode),
                        }
                        progress_bar.set_postfix(
                            loss=round(logs["loss"], 4),
                            lr=round(logs["lr"], 8),
                        )
                        if accelerator.is_main_process:
                            append_jsonl(train_metric_file, logs)

                        _maybe_run_checkpoint_and_validation(epoch, step_in_epoch, global_step)

                    if global_step >= args.max_train_steps:
                        break

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

    # finalize
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter_state = get_trainable_state_dict(accelerator, unet_xs)
        adapter_path = os.path.join(args.output_dir, _ADAPTER_WEIGHTS_FILE)
        torch.save(adapter_state, adapter_path)
        logger.info(f"Saved final adapter weights to {adapter_path}")

    accelerator.wait_for_everyone()

    if args.validation_h5_paths:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            log_validation_h5_xs_1ch_multi(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet_xs=unet_xs,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        removed = 0
        for root, _, files in os.walk(args.output_dir):
            for fname in files:
                if fname in ("model.safetensors", "pytorch_model.bin", "diffusion_pytorch_model.safetensors"):
                    fpath = os.path.join(root, fname)
                    try:
                        os.remove(fpath)
                        removed += 1
                        logger.info(f"Removed large legacy checkpoint file: {fpath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {fpath}: {e}")
        if removed > 0:
            logger.info(f"Cleanup done, removed {removed} legacy weight files.")

    accelerator.end_training()


if __name__ == "__main__":
    args = load_config()
    main(args)
