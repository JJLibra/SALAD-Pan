import argparse
import ast
import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import check_min_version

from core.components.salad_pan import DualBranchXSAdapter, UNetDualBranchXSModel
from core.pipelines.salad_pan import StableDiffusionDualBranchXSPipeline
from utils.metrics import Q4_numpy, Q8_numpy, SAM_numpy, ERGAS_numpy, SCC_numpy

check_min_version("0.36.0.dev0")
logger = logging.getLogger(__name__)

_LEGACY_CN = "control" + "net"
_LEGACY_CNX = _LEGACY_CN + "_xs"
_SSIM_KERNEL_CACHE: Dict[tuple, torch.Tensor] = {}


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
        raise KeyError("Missing required config keys:\n" + "\n".join([f"  - {k}" for k in missing]))


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_image_safe(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def sanitize_name(name: str) -> str:
    s = str(name)
    for a, b in [(" ", "_"), ("/", "_"), ("\\", "_"), (":", "_")]:
        s = s.replace(a, b)
    return s


def _legacy_key(suffix: str) -> str:
    return f"{_LEGACY_CN}_{suffix}"


def _legacy_xs_key(suffix: str) -> str:
    return f"{_LEGACY_CNX}_{suffix}"


def load_config():
    ap = argparse.ArgumentParser(description="SALAD-Pan diffusion inference for H5 datasets", add_help=True)
    ap.add_argument("--config", type=str, required=True, help="YAML config file path")
    ap.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help='Override fields in YAML, e.g.: -o num_inference_steps=30 -o "inference_indices=[0,1]"',
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

    cfg["input_h5_paths"] = _as_list(cfg.get("input_h5_paths"))
    cfg["input_h5_names"] = _as_list(cfg.get("input_h5_names"))

    required_keys = [
        "pretrained_model_name_or_path",
        "vae_path",
        "output_dir",
        "local_files_only",
        "revision",
        "variant",
        "tokenizer_name",
        "seed",
        "device",
        "mixed_precision",
        "unet_adapter_model_name_or_path",
        "adapter_weights_path",
        "adapter_size_ratio",
        "adapter_learn_time_embedding",
        "adapter_time_embedding_mix",
        "conditioning_scale_spa",
        "conditioning_scale_spe",
        "input_h5_paths",
        "input_h5_names",
        "h5_keys",
        "resolution",
        "range_clip_min",
        "range_clip_max",
        "range_clip_max_map",
        "discard_out_of_range",
        "dataset_prompts",
        "band_prompts",
        "use_prompts_in_inference",
        "num_inference_steps",
        "guidance_scale",
        "infer_band_batch_size",
        "inference_count",
        "inference_indices",
        "save_pred_h5",
        "save_visual_rgb",
        "save_metrics_jsonl",
    ]
    _require_keys(cfg, required_keys)

    if cfg["input_h5_paths"] is None or len(cfg["input_h5_paths"]) == 0:
        raise ValueError("`input_h5_paths` must contain at least one path.")
    if cfg["input_h5_names"] is None or len(cfg["input_h5_names"]) != len(cfg["input_h5_paths"]):
        raise ValueError("`input_h5_names` must match `input_h5_paths` length.")
    if int(cfg["resolution"]) % 8 != 0:
        raise ValueError("`resolution` must be divisible by 8.")
    if not isinstance(cfg["h5_keys"], dict) or any(k not in cfg["h5_keys"] for k in ("gt", "lms", "pan")):
        raise ValueError("`h5_keys` must contain `gt`, `lms`, `pan`.")

    if cfg["local_files_only"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

    for p in cfg["input_h5_paths"]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input H5 not found: {p}")

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


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int, device, dtype):
    key = (kernel_size, sigma, channels, str(device), str(dtype))
    if key in _SSIM_KERNEL_CACHE:
        return _SSIM_KERNEL_CACHE[key]
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    g2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    kernel = g2d.repeat(channels, 1, 1, 1)
    _SSIM_KERNEL_CACHE[key] = kernel
    return kernel


def calc_mse(x, y):
    return F.mse_loss(x, y, reduction="mean")


def calc_mae(x, y):
    return F.l1_loss(x, y, reduction="mean")


def calc_psnr(x01, y01, eps=1e-10):
    mse = calc_mse(x01, y01).clamp(min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def calc_ssim(x01, y01, kernel_size=11, sigma=1.5):
    _, c, _, _ = x01.shape
    kernel = _gaussian_kernel(kernel_size, sigma, c, x01.device, x01.dtype)
    pad = kernel_size // 2
    C1, C2 = 0.01**2, 0.03**2

    mu_x = F.conv2d(x01, kernel, groups=c, padding=pad)
    mu_y = F.conv2d(y01, kernel, groups=c, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sig_x2 = F.conv2d(x01 * x01, kernel, groups=c, padding=pad) - mu_x2
    sig_y2 = F.conv2d(y01 * y01, kernel, groups=c, padding=pad) - mu_y2
    sig_xy = F.conv2d(x01 * y01, kernel, groups=c, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return ssim_map.mean()


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


def _torch_load_compat(path: str, *, map_location="cpu", weights_only: Optional[bool] = None):
    sig = torch.load.__code__.co_varnames
    kwargs = {"map_location": map_location}
    if weights_only is not None and "weights_only" in sig:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def _select_weight_dtype(mixed_precision: str):
    mp = str(mixed_precision).lower().strip()
    if mp == "fp16":
        return torch.float16
    if mp == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_indices(args, ds_name: str, N: int):
    specified_idx = args.inference_indices
    count = int(args.inference_count)
    if isinstance(specified_idx, dict):
        idxs = specified_idx.get(ds_name, None)
        if idxs is None:
            idxs = specified_idx.get(sanitize_name(ds_name), None)
        if idxs is not None:
            idxs = list(idxs)
        else:
            rng = np.random.default_rng(args.seed)
            idxs = rng.choice(N, size=min(count, N), replace=False).tolist()
    elif specified_idx is None:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(N, size=min(count, N), replace=False).tolist()
    else:
        idxs = list(specified_idx)

    idxs = [int(i) for i in idxs if 0 <= int(i) < N]
    if len(idxs) == 0:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(N, size=min(count, N), replace=False).tolist()
    if count > 0:
        idxs = idxs[: min(count, len(idxs))]
    return idxs


def run_inference(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    weight_dtype = _select_weight_dtype(args.mixed_precision)
    autocast_ctx = contextlib.nullcontext()
    if device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast("cuda", dtype=weight_dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_inference_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=False)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.pretrained_model_name_or_path
    tokenizer_kwargs = dict(revision=args.revision, use_fast=False, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        **({"subfolder": "tokenizer"} if tokenizer_name == args.pretrained_model_name_or_path else {}),
        **tokenizer_kwargs,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, args.local_files_only
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=args.local_files_only)
    base_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    scheduler = UniPCMultistepScheduler.from_config(base_scheduler.config)

    if args.unet_adapter_model_name_or_path:
        logger.info("Loading UNetDualBranchXSModel from pretrained path.")
        unet = UNetDualBranchXSModel.from_pretrained(
            args.unet_adapter_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            local_files_only=args.local_files_only,
        )
    else:
        logger.info("Building UNetDualBranchXSModel from base UNet + adapter template.")
        base_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
            local_files_only=args.local_files_only,
        )
        adapter = DualBranchXSAdapter.from_unet(
            base_unet,
            size_ratio=float(args.adapter_size_ratio),
            learn_time_embedding=bool(args.adapter_learn_time_embedding),
            time_embedding_mix=float(args.adapter_time_embedding_mix),
            conditioning_channels=5,
            conditioning_channel_order="rgb",
        )
        unet = UNetDualBranchXSModel.from_unet(base_unet, adapter=adapter)
        del base_unet
        del adapter

    if args.adapter_weights_path:
        awp = Path(args.adapter_weights_path)
        if not awp.exists():
            raise FileNotFoundError(f"adapter_weights_path not found: {awp}")
        logger.info(f"Loading adapter weights from: {awp}")
        adapter_state = _torch_load_compat(str(awp), map_location="cpu", weights_only=True)
        missing, unexpected = unet.load_state_dict(adapter_state, strict=False)
        logger.info(f"Adapter weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    pipe = StableDiffusionDualBranchXSPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        adapter=None,
        **{_LEGACY_CN: None},
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    metrics_path = output_dir / "inference_metrics.jsonl"
    vis_root = output_dir / "inference_vis"
    pred_root = output_dir / "inference_h5"

    def _resize_np(x, size_hw):
        t = torch.from_numpy(x[None])
        t = F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)
        return t[0].numpy()

    for h5_path, ds_name in zip(args.input_h5_paths, args.input_h5_names):
        clip_min = float(args.range_clip_min)
        clip_max = resolve_clip_max(args, ds_name, h5_path)
        ds_prompt = resolve_dataset_prompt(args, ds_name, h5_path) if bool(args.use_prompts_in_inference) else ""
        band_prompts = resolve_band_prompts(args, ds_name, h5_path) if bool(args.use_prompts_in_inference) else None
        ds_tag = sanitize_name(ds_name)

        logger.info(f"[Inference] dataset={ds_name}, file={Path(h5_path).name}, clip_max={clip_max}")

        with h5py.File(h5_path, "r") as f:
            gt_ds = f[args.h5_keys["gt"]]
            lms_ds = f[args.h5_keys["lms"]]
            pan_ds = f[args.h5_keys["pan"]]
            N = int(gt_ds.shape[0])
            idxs = _resolve_indices(args, ds_name, N)

            pred_list = []
            idx_list = []
            sample_metrics = []

            for order, idx in enumerate(idxs):
                gtC = np.array(gt_ds[idx], dtype=np.float32)
                lmsC = np.array(lms_ds[idx], dtype=np.float32)
                pan = np.array(pan_ds[idx], dtype=np.float32)

                if bool(args.discard_out_of_range):
                    mn = min(gtC.min(), lmsC.min(), pan.min())
                    mx = max(gtC.max(), lmsC.max(), pan.max())
                    if (mn < clip_min) or (mx > clip_max):
                        logger.info(
                            f"[Inference:{ds_name}] skip idx={idx} out-of-range [{mn:.1f},{mx:.1f}] not in [{clip_min},{clip_max}]"
                        )
                        continue

                gtC = np.clip(gtC, clip_min, clip_max) / clip_max
                lmsC = np.clip(lmsC, clip_min, clip_max) / clip_max
                pan = np.clip(pan, clip_min, clip_max) / clip_max

                H, W = gtC.shape[-2:]
                if (H != int(args.resolution)) or (W != int(args.resolution)):
                    gtC = _resize_np(gtC, (int(args.resolution), int(args.resolution)))
                    lmsC = _resize_np(lmsC, (int(args.resolution), int(args.resolution)))
                    pan = _resize_np(pan, (int(args.resolution), int(args.resolution)))

                C = int(gtC.shape[0])
                band_bs = int(args.infer_band_batch_size)
                gen_bands = []

                for start in range(0, C, band_bs):
                    end = min(C, start + band_bs)
                    cond_list, prompts = [], []

                    for c in range(start, end):
                        lms_band = lmsC[c : c + 1]
                        cond5 = np.concatenate([np.repeat(lms_band, 4, axis=0), pan], axis=0)
                        cond_list.append(cond5)
                        if bool(args.use_prompts_in_inference):
                            bp = band_prompts[c] if (band_prompts is not None and c < len(band_prompts)) else None
                            prompts.append(build_prompt_text(ds_prompt, bp))
                        else:
                            prompts.append("")

                    cond_t = torch.from_numpy(np.stack(cond_list, axis=0)).to(device=device, dtype=weight_dtype)
                    with autocast_ctx:
                        out = pipe(
                            prompt=prompts,
                            image=cond_t,
                            num_inference_steps=int(args.num_inference_steps),
                            guidance_scale=float(args.guidance_scale),
                            generator=generator,
                            output_type="pt",
                            conditioning_scale=1.0,
                            conditioning_scale_spa=float(args.conditioning_scale_spa),
                            conditioning_scale_spe=float(args.conditioning_scale_spe),
                        )
                    gen_band_01 = out.images
                    if not torch.is_tensor(gen_band_01):
                        gen_band_01 = torch.stack(gen_band_01, dim=0)
                    gen_bands.append(gen_band_01.to(device=device, dtype=torch.float32))

                gen_band_01 = torch.cat(gen_bands, dim=0)
                pred_ms_01 = gen_band_01[:, 0, :, :].unsqueeze(0)
                gt_ms_01 = torch.from_numpy(gtC).unsqueeze(0).to(device=device, dtype=torch.float32)

                mse_i = float(calc_mse(pred_ms_01, gt_ms_01).item())
                mae_i = float(calc_mae(pred_ms_01, gt_ms_01).item())
                psnr_i = float(calc_psnr(pred_ms_01, gt_ms_01).item())
                ssim_i = float(calc_ssim(pred_ms_01, gt_ms_01).item())

                pred_np = pred_ms_01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                gt_np = gt_ms_01[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

                if C == 8:
                    q_name, q_fn = "q8", Q8_numpy
                else:
                    q_name, q_fn = "q4", Q4_numpy
                try:
                    q_i = float(q_fn(gt_np, pred_np))
                except Exception:
                    q_i = float("nan")
                try:
                    sam_i = float(SAM_numpy(gt_np, pred_np))
                except Exception:
                    sam_i = float("nan")
                try:
                    ergas_i = float(ERGAS_numpy(gt_np, pred_np))
                except Exception:
                    ergas_i = float("nan")
                try:
                    scc_i = float(SCC_numpy(gt_np, pred_np))
                except Exception:
                    scc_i = float("nan")

                metric_record = {
                    "time": datetime.now().isoformat(),
                    "dataset": ds_name,
                    "dataset_idx": int(idx),
                    "sample_order": int(order),
                    "mse": mse_i,
                    "mae": mae_i,
                    "psnr": psnr_i,
                    "ssim": ssim_i,
                    q_name: q_i,
                    "sam": sam_i,
                    "ergas": ergas_i,
                    "scc": scc_i,
                }
                sample_metrics.append(metric_record)
                if bool(args.save_metrics_jsonl):
                    append_jsonl(metrics_path, metric_record)

                if bool(args.save_visual_rgb):
                    try:
                        if pred_ms_01.shape[1] >= 4:
                            gt_rgb = to_pil_image(gt_ms_01[0, 1:4].detach().cpu())
                            pr_rgb = to_pil_image(pred_ms_01[0, 1:4].detach().cpu())
                        elif pred_ms_01.shape[1] >= 3:
                            gt_rgb = to_pil_image(gt_ms_01[0, :3].detach().cpu())
                            pr_rgb = to_pil_image(pred_ms_01[0, :3].detach().cpu())
                        else:
                            gt_rgb = to_pil_image(gt_ms_01[0, 0:1].repeat(3, 1, 1).detach().cpu())
                            pr_rgb = to_pil_image(pred_ms_01[0, 0:1].repeat(3, 1, 1).detach().cpu())
                        sample_dir = vis_root / ds_tag / f"{int(order):04d}_idx{int(idx)}"
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        save_image_safe(gt_rgb, sample_dir / "gt_rgb.png")
                        save_image_safe(pr_rgb, sample_dir / "pred_rgb.png")
                    except Exception as e:
                        logger.warning(f"[Inference:{ds_name}] failed to save RGB at idx={idx}: {e}")

                pred_list.append(pred_ms_01[0].detach().cpu().numpy())
                idx_list.append(int(idx))

        if bool(args.save_pred_h5) and len(pred_list) > 0:
            pred_root.mkdir(parents=True, exist_ok=True)
            save_path = pred_root / f"{ds_tag}_pred.h5"
            pred_arr = np.stack(pred_list, axis=0).astype(np.float32)  # [N,C,H,W] in [0,1]
            with h5py.File(save_path, "w") as wf:
                wf.create_dataset("pred_01", data=pred_arr, compression="gzip")
                wf.create_dataset("pred_native", data=(pred_arr * clip_max).astype(np.float32), compression="gzip")
                wf.create_dataset("indices", data=np.array(idx_list, dtype=np.int32))
            logger.info(f"[Inference] saved predictions: {save_path}")

        if bool(args.save_metrics_jsonl) and len(sample_metrics) > 0:
            q_key = "q8" if any("q8" in r for r in sample_metrics) else "q4"
            summary = {
                "time": datetime.now().isoformat(),
                "dataset": ds_name,
                "sample_kind": f"mean_{len(sample_metrics)}",
                "mean_mse": float(np.nanmean([r["mse"] for r in sample_metrics])),
                "mean_mae": float(np.nanmean([r["mae"] for r in sample_metrics])),
                "mean_psnr": float(np.nanmean([r["psnr"] for r in sample_metrics])),
                "mean_ssim": float(np.nanmean([r["ssim"] for r in sample_metrics])),
                f"mean_{q_key}": float(np.nanmean([r.get(q_key, np.nan) for r in sample_metrics])),
                "mean_sam": float(np.nanmean([r["sam"] for r in sample_metrics])),
                "mean_ergas": float(np.nanmean([r["ergas"] for r in sample_metrics])),
                "mean_scc": float(np.nanmean([r["scc"] for r in sample_metrics])),
            }
            append_jsonl(metrics_path, summary)
            logger.info(
                f"[Inference] dataset={ds_name} mean_psnr={summary['mean_psnr']:.4f}, "
                f"mean_ssim={summary['mean_ssim']:.4f}, mean_{q_key}={summary[f'mean_{q_key}']:.4f}"
            )


def main():
    args = load_config()
    run_inference(args)


if __name__ == "__main__":
    main()
