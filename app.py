from __future__ import annotations

import contextlib
import importlib.util
import os
import random
import sys
import tarfile
import tempfile
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import gradio as gr
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    PretrainedConfig,
)

cudnn.benchmark = True
cudnn.deterministic = False

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference.yaml"

BASE_MODEL_CHOICES = {
    "stable-diffusion-v1-5": "base/stable-diffusion-v1-5",
    "stable-diffusion-v2": "base/stable-diffusion-v2",
    "stable-diffusion-xl": "base/stable-diffusion-xl",
}
CUSTOM_BASE_KEY = "Custom upload"

DATASET_PRESETS = {
    "Use config values": {},
    "Preset-A": {
        "vae_path": "checkpoints/vae_c1",
        "dual_branch_model_name_or_path": "checkpoints/diffusion",
        "test_h5_path": "data/preset_a/test.h5",
    },
    "Preset-B": {
        "vae_path": "checkpoints/vae_c1",
        "dual_branch_model_name_or_path": "checkpoints/diffusion",
        "test_h5_path": "data/preset_b/test.h5",
    },
    "Preset-C": {
        "vae_path": "checkpoints/vae_c1",
        "dual_branch_model_name_or_path": "checkpoints/diffusion",
        "test_h5_path": "data/preset_c/test.h5",
    },
}


def _resolve_path(path_like: str | Path | None) -> str | None:
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve(strict=False))


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_existing_file(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the candidate files exist:\n" + "\n".join(str(p) for p in candidates)
    )


def _pick_attr(module: Any, candidates: tuple[str, ...], module_label: str):
    for name in candidates:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AttributeError(f"Could not find any of {candidates} in {module_label}")


def _load_project_modules():
    inference_file = PROJECT_ROOT / "inference.py"
    metrics_file = PROJECT_ROOT / "utils" / "metrics.py"

    model_file = _first_existing_file(
        [
            PROJECT_ROOT / "core" / "models" / "controlnet_xs_spa_spe_c1_glu.py",
            PROJECT_ROOT / "core" / "components" / "cc_pan.py",
            PROJECT_ROOT / "core" / "components" / "cc-pan.py",
        ]
    )
    pipeline_file = _first_existing_file(
        [
            PROJECT_ROOT / "core" / "pipelines" / "cc_pan.py",
            PROJECT_ROOT / "core" / "pipelines" / "cc-pan.py",
        ]
    )

    inference_module = (
        _load_module_from_file("project_inference", inference_file)
        if inference_file.exists()
        else None
    )
    metrics_module = _load_module_from_file("project_metrics", metrics_file)
    model_module = _load_module_from_file("project_dual_branch_model", model_file)
    pipeline_module = _load_module_from_file("project_dual_branch_pipeline", pipeline_file)

    return inference_module, metrics_module, model_module, pipeline_module


INFERENCE_MODULE, METRICS_MODULE, MODEL_MODULE, PIPELINE_MODULE = _load_project_modules()

DualBranchAdapterClass = _pick_attr(
    MODEL_MODULE,
    (
        "DualBranchXSAdapter",
    ),
    "model module",
)

UNetDualBranchXSModelClass = _pick_attr(
    MODEL_MODULE,
    (
        "UNetDualBranchXSModel",
    ),
    "model module",
)

DualBranchPipelineClass = _pick_attr(
    PIPELINE_MODULE,
    (
        "StableDiffusionDualBranchXSPipeline",
    ),
    "pipeline module",
)

psnr = _pick_attr(METRICS_MODULE, ("psnr",), "utils/metrics.py")
ssim = _pick_attr(METRICS_MODULE, ("ssim",), "utils/metrics.py")
sam_deg_torch = _pick_attr(METRICS_MODULE, ("sam_deg_torch",), "utils/metrics.py")
_hqnr = _pick_attr(METRICS_MODULE, ("_hqnr", "hqnr"), "utils/metrics.py")
Q4_numpy = _pick_attr(METRICS_MODULE, ("Q4_numpy",), "utils/metrics.py")
SAM_numpy = _pick_attr(METRICS_MODULE, ("SAM_numpy",), "utils/metrics.py")
ERGAS_numpy = _pick_attr(METRICS_MODULE, ("ERGAS_numpy",), "utils/metrics.py")
SCC_numpy = _pick_attr(METRICS_MODULE, ("SCC_numpy",), "utils/metrics.py")
CC_numpy = _pick_attr(METRICS_MODULE, ("CC_numpy",), "utils/metrics.py")


def _fallback_resize_np_chw(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(array).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()


def _fallback_resize_hwc01(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    chw = np.transpose(array, (2, 0, 1)).astype(np.float32)
    chw = _fallback_resize_np_chw(chw, size)
    return np.transpose(chw, (1, 2, 0))


def _fallback_to_hwc01(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.permute(1, 2, 0).numpy()
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4, 5, 8}:
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


_resize_np_chw = getattr(METRICS_MODULE, "_resize_np_chw", _fallback_resize_np_chw)
_resize_hwc01 = getattr(METRICS_MODULE, "_resize_hwc01", _fallback_resize_hwc01)
_to_hwc01 = getattr(METRICS_MODULE, "_to_hwc01", _fallback_to_hwc01)


def _default_config_dict() -> dict[str, Any]:
    return {
        "pretrained_model_name_or_path": "",
        "vae_path": "",
        "dual_branch_model_name_or_path": "",
        "test_h5_path": "",
        "local_files_only": True,
        "eval_mode": "reduced",
        "mixed_precision": "fp16",
        "seed": 2025,
        "resolution": 1024,
        "range_clip_min": 0.0,
        "range_clip_max": 2047.0,
        "discard_out_of_range": False,
        "ergas_scale": 4.0,
        "controlnet_conditioning_scale": 1.0,
        "controlnet_conditioning_scale_spa": 1.0,
        "controlnet_conditioning_scale_spe": 1.0,
        "guidance_scale": 1.0,
        "test_num_inference_steps": 50,
        "enable_xformers_memory_efficient_attention": False,
        "enable_vae_tiling": False,
        "enable_vae_slicing": False,
        "enable_attention_slicing": False,
        "h5_keys": {
            "gt": "gt",
            "lms": "lms",
            "pan": "pan",
        },
    }


def _dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    return obj


def _namespace_to_dict(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_namespace_to_dict(v) for v in ns]
    return ns


def _load_config_from_yaml(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    defaults = _default_config_dict()
    defaults.update(config_data)

    if "h5_keys" not in defaults or defaults["h5_keys"] is None:
        defaults["h5_keys"] = _default_config_dict()["h5_keys"]

    if "dual_branch_model_name_or_path" not in defaults:
        defaults["dual_branch_model_name_or_path"] = config_data.get(
            "controlnet_model_name_or_path", ""
        )

    if (
        "controlnet_conditioning_scale_spa" not in defaults
        or defaults["controlnet_conditioning_scale_spa"] is None
    ):
        defaults["controlnet_conditioning_scale_spa"] = defaults["controlnet_conditioning_scale"]

    if (
        "controlnet_conditioning_scale_spe" not in defaults
        or defaults["controlnet_conditioning_scale_spe"] is None
    ):
        defaults["controlnet_conditioning_scale_spe"] = defaults["controlnet_conditioning_scale"]

    return _dict_to_namespace(defaults)


def load_config_for_demo(config_path: str | Path):
    if INFERENCE_MODULE is not None:
        external_load_config = getattr(INFERENCE_MODULE, "load_config", None)
        if callable(external_load_config):
            try:
                return external_load_config(str(config_path))
            except TypeError:
                old_argv = sys.argv
                try:
                    sys.argv = [old_argv[0], "--config", str(config_path)]
                    return external_load_config()
                except Exception:
                    traceback.print_exc()
                finally:
                    sys.argv = old_argv
            except Exception:
                traceback.print_exc()

    return _load_config_from_yaml(config_path)


def import_text_encoder_class(
    pretrained_model_name_or_path: str,
    local_files_only: bool = True,
):
    if INFERENCE_MODULE is not None:
        external_import = getattr(INFERENCE_MODULE, "import_text_encoder_class", None)
        if callable(external_import):
            try:
                return external_import(
                    pretrained_model_name_or_path,
                    local_files_only=local_files_only,
                )
            except Exception:
                traceback.print_exc()

    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        local_files_only=local_files_only,
    )

    architectures = getattr(text_encoder_config, "architectures", None) or []
    architecture = architectures[0] if architectures else "CLIPTextModel"

    if architecture == "CLIPTextModel":
        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection

    raise ValueError(f"Unsupported text encoder architecture: {architecture}")


def _is_safe_extract_target(base_dir: Path, target_path: Path) -> bool:
    base_resolved = base_dir.resolve(strict=False)
    target_resolved = target_path.resolve(strict=False)
    return os.path.commonpath([str(base_resolved), str(target_resolved)]) == str(base_resolved)


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target_path = extract_dir / member.filename
            if not _is_safe_extract_target(extract_dir, target_path):
                raise ValueError(f"Unsafe zip member path detected: {member.filename}")
        zf.extractall(extract_dir)


def _safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            target_path = extract_dir / member.name
            if not _is_safe_extract_target(extract_dir, target_path):
                raise ValueError(f"Unsafe tar member path detected: {member.name}")
        tf.extractall(extract_dir)


def _find_model_root(path: Path) -> Path:
    if not path.exists():
        return path

    if path.is_file():
        return path.parent

    if (path / "model_index.json").exists():
        return path

    expected_subdirs = {"tokenizer", "text_encoder", "unet", "scheduler"}
    if all((path / subdir).exists() for subdir in expected_subdirs):
        return path

    for candidate in path.rglob("model_index.json"):
        return candidate.parent

    children = [p for p in path.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(children) == 1:
        return _find_model_root(children[0])

    return path


def _prepare_custom_base_model(upload_path: str | Path) -> str:
    upload_path = Path(upload_path)

    if upload_path.is_dir():
        return str(_find_model_root(upload_path))

    extract_dir = Path(tempfile.mkdtemp(prefix="base_model_"))

    try:
        if zipfile.is_zipfile(upload_path):
            _safe_extract_zip(upload_path, extract_dir)
            return str(_find_model_root(extract_dir))

        if tarfile.is_tarfile(upload_path):
            _safe_extract_tar(upload_path, extract_dir)
            return str(_find_model_root(extract_dir))
    except Exception as exc:
        print(f"[WARN] Failed to unpack custom base model: {exc}. Fallback to parent directory.")

    return str(_find_model_root(upload_path.parent))


def _unwrap_uploaded_path(upload_path: Any) -> str | None:
    if upload_path is None:
        return None
    if isinstance(upload_path, list):
        return str(upload_path[0]) if upload_path else None
    return str(upload_path)


def _to_plain_dict(ns_or_dict: Any) -> dict[str, Any]:
    if isinstance(ns_or_dict, dict):
        return ns_or_dict
    return _namespace_to_dict(ns_or_dict)


class DualBranchFusionDemoRunner:
    def __init__(self, config_path: str, overrides: dict[str, Any] | None = None):
        args = load_config_for_demo(config_path)
        args_dict = _to_plain_dict(args)

        if overrides:
            args_dict.update(overrides)

        if "h5_keys" not in args_dict or args_dict["h5_keys"] is None:
            args_dict["h5_keys"] = _default_config_dict()["h5_keys"]

        if "dual_branch_model_name_or_path" not in args_dict:
            args_dict["dual_branch_model_name_or_path"] = args_dict.get("controlnet_model_name_or_path", "")

        if (
            "controlnet_conditioning_scale_spa" not in args_dict
            or args_dict["controlnet_conditioning_scale_spa"] is None
        ):
            args_dict["controlnet_conditioning_scale_spa"] = args_dict.get("controlnet_conditioning_scale", 1.0)

        if (
            "controlnet_conditioning_scale_spe" not in args_dict
            or args_dict["controlnet_conditioning_scale_spe"] is None
        ):
            args_dict["controlnet_conditioning_scale_spe"] = args_dict.get("controlnet_conditioning_scale", 1.0)

        self.args = _dict_to_namespace(args_dict)
        self.mode = str(self.args.eval_mode).lower().strip()
        self.is_reduced = self.mode == "reduced"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_dtype, self.autocast_dtype = self._resolve_precision()
        self.empty_prompt_embeds: torch.Tensor | None = None
        self.h5: h5py.File | None = None

        self._set_seed()
        self._load_models()
        self._configure_pipeline()
        self._cache_empty_prompt_embeds()
        self._open_dataset()
        self._load_runtime_settings()

    def _resolve_precision(self) -> tuple[torch.dtype, torch.dtype | None]:
        precision = str(self.args.mixed_precision).lower()
        if precision == "fp16" and self.device.type == "cuda":
            return torch.float16, torch.float16
        if precision == "bf16" and self.device.type == "cuda":
            return torch.bfloat16, torch.bfloat16
        return torch.float32, None

    def _autocast_context(self):
        if self.device.type == "cuda" and self.autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _set_seed(self) -> None:
        if self.args.seed is None:
            return
        seed = int(self.args.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _load_models(self) -> None:
        base_model_path = self.args.pretrained_model_name_or_path
        local_files_only = bool(self.args.local_files_only)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
            local_files_only=local_files_only,
            use_fast=False,
        )

        text_encoder_class = import_text_encoder_class(
            base_model_path,
            local_files_only=local_files_only,
        )

        self.text_encoder = text_encoder_class.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            local_files_only=local_files_only,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.args.vae_path,
            local_files_only=local_files_only,
        )

        self.dual_branch_adapter = DualBranchAdapterClass.from_pretrained(
            self.args.dual_branch_model_name_or_path,
            local_files_only=local_files_only,
        )

        scheduler = DDPMScheduler.from_pretrained(
            base_model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        scheduler = UniPCMultistepScheduler.from_config(scheduler.config)

        self.pipe = DualBranchPipelineClass(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.dual_branch_adapter,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)

    def _configure_pipeline(self) -> None:
        try:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
            if getattr(self.pipe, "controlnet", None) is not None:
                self.pipe.controlnet.to(memory_format=torch.channels_last)
        except Exception as exc:
            print(f"[WARN] Failed to set channels_last: {exc}")

        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        if getattr(self.args, "enable_xformers_memory_efficient_attention", False):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(f"[WARN] Failed to enable xformers attention: {exc}")

        if getattr(self.args, "enable_vae_tiling", False):
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                try:
                    self.pipe.vae.enable_tiling()
                except Exception as exc:
                    print(f"[WARN] Failed to enable VAE tiling: {exc}")

        if getattr(self.args, "enable_vae_slicing", False):
            try:
                self.pipe.vae.enable_slicing()
            except Exception as exc:
                print(f"[WARN] Failed to enable VAE slicing: {exc}")

        if getattr(self.args, "enable_attention_slicing", False):
            try:
                self.pipe.enable_attention_slicing("max")
            except Exception as exc:
                print(f"[WARN] Failed to enable attention slicing: {exc}")

    def _cache_empty_prompt_embeds(self) -> None:
        empty_prompt = [""]

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=empty_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        self.empty_prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype, device=self.device)

    def _open_dataset(self) -> None:
        self.h5 = h5py.File(self.args.test_h5_path, "r")
        h5_keys = self.args.h5_keys if isinstance(self.args.h5_keys, dict) else vars(self.args.h5_keys)

        self.gt_ds = self.h5[h5_keys["gt"]] if self.is_reduced else None
        self.lms_ds = self.h5[h5_keys["lms"]]
        self.pan_ds = self.h5[h5_keys["pan"]]

        self.N = self.gt_ds.shape[0] if self.is_reduced else self.lms_ds.shape[0]

    def _load_runtime_settings(self) -> None:
        self.res = int(self.args.resolution)
        self.clip_min = float(self.args.range_clip_min)
        self.clip_max = float(self.args.range_clip_max)
        self.drop_oob = bool(self.args.discard_out_of_range)
        self.r_ergas = float(self.args.ergas_scale)

        self.gen = (
            None
            if self.args.seed is None
            else torch.Generator(device=self.device).manual_seed(int(self.args.seed))
        )

    def close(self) -> None:
        if self.h5 is not None:
            try:
                self.h5.close()
            except Exception:
                pass
            self.h5 = None

    def __del__(self):
        self.close()

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.N:
            raise ValueError(f"idx must be in [0, {self.N - 1}]")

    def _clip_and_normalize(self, array: np.ndarray) -> np.ndarray:
        return np.clip(array, self.clip_min, self.clip_max) / self.clip_max

    def _resize_chw_if_needed(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-2:] == (self.res, self.res):
            return array
        return _resize_np_chw(array, (self.res, self.res))

    def _get_sample(self, idx: int) -> dict[str, np.ndarray | None]:
        self._validate_index(idx)

        gt = np.array(self.gt_ds[idx], dtype=np.float32) if self.is_reduced else None
        lms = np.array(self.lms_ds[idx], dtype=np.float32)
        pan = np.array(self.pan_ds[idx], dtype=np.float32)

        if self.drop_oob:
            arrays = [lms, pan] + ([gt] if gt is not None else [])
            vmin = min(arr.min() for arr in arrays)
            vmax = max(arr.max() for arr in arrays)
            if vmin < self.clip_min or vmax > self.clip_max:
                raise ValueError(
                    f"Sample {idx} has value range [{vmin:.1f}, {vmax:.1f}], "
                    f"which is outside [{self.clip_min}, {self.clip_max}] and must be discarded."
                )

        gt = self._clip_and_normalize(gt) if gt is not None else None
        lms = self._clip_and_normalize(lms)
        pan = self._clip_and_normalize(pan)

        return {
            "gt": gt,
            "gt_res": self._resize_chw_if_needed(gt) if gt is not None else None,
            "lms": lms,
            "lms_res_for_cond": self._resize_chw_if_needed(lms),
            "pan": pan,
            "pan_res": self._resize_chw_if_needed(pan),
        }

    @staticmethod
    def _lms_to_rgb(lms_chw: np.ndarray):
        rgb_channels = min(3, lms_chw.shape[0])
        return to_pil_image(torch.from_numpy(lms_chw[:rgb_channels]).clamp(0, 1))

    @staticmethod
    def _pan_to_rgb(pan_chw: np.ndarray):
        pan_rgb = np.repeat(pan_chw[0:1], 3, axis=0)
        return to_pil_image(torch.from_numpy(pan_rgb).clamp(0, 1))

    def preview_inputs(self, idx: int):
        sample = self._get_sample(idx)
        lms_img = self._lms_to_rgb(sample["lms_res_for_cond"])
        pan_img = self._pan_to_rgb(sample["pan_res"])
        return lms_img, pan_img

    def _compute_reference_metrics(
        self,
        gen01: torch.Tensor,
        gt01: torch.Tensor,
    ) -> tuple[dict[str, float | str], np.ndarray]:
        metrics: dict[str, float | str] = {}
        gen_hwc = _to_hwc01(gen01)
        gt_hwc = _to_hwc01(gt01)

        metrics.update(
            {
                "PSNR": float(psnr(gen01, gt01).item()),
                "SSIM": float(ssim(gen01, gt01).item()),
                "SAM_torch_deg": float(sam_deg_torch(gen01, gt01).item()),
            }
        )

        try:
            num_bands = gen_hwc.shape[-1]
            if num_bands == 4:
                metrics["Q4"] = float(Q4_numpy(gt_hwc, gen_hwc))
            metrics.update(
                {
                    "SAM_numpy_deg": float(SAM_numpy(gt_hwc, gen_hwc)),
                    "ERGAS": float(ERGAS_numpy(gt_hwc, gen_hwc, ratio=(1.0 / self.r_ergas))),
                    "SCC": float(SCC_numpy(gt_hwc, gen_hwc)),
                    "CC": float(CC_numpy(gt_hwc, gen_hwc)),
                }
            )
        except AssertionError as exc:
            print("=== Error while computing NumPy reference metrics ===")
            traceback.print_exc()
            metrics["metric_error"] = f"NumPy metrics assertion: {exc!r}"

        return metrics, gen_hwc

    def _compute_no_reference_metrics(
        self,
        lms: np.ndarray,
        pan_res: np.ndarray,
        gen_hwc: np.ndarray,
    ) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {}

        pan_for_ds = pan_res[0] if pan_res.ndim == 3 else pan_res.squeeze(0)
        lp_size = (pan_for_ds.shape[0] // 4, pan_for_ds.shape[1] // 4)

        lms_for_ds = np.transpose(lms, (1, 2, 0))
        if lms_for_ds.shape[:2] != lp_size:
            lms_for_ds = _resize_hwc01(lms_for_ds, lp_size)

        try:
            hqnr_val, d_lambda, d_s = _hqnr(lms_for_ds, pan_for_ds, gen_hwc)
            metrics.update(
                {
                    "D_lambda": float(d_lambda),
                    "D_s": float(d_s),
                    "HQNR": float(hqnr_val),
                }
            )
        except AssertionError as exc:
            print("=== Error while computing HQNR / D_lambda / D_s ===")
            traceback.print_exc()
            metrics["hqnr_error"] = f"HQNR metrics assertion: {exc!r}"

        return metrics

    def infer_one_stream(self, idx: int) -> Iterator[tuple[Any, dict[str, Any]]]:
        with torch.no_grad():
            sample = self._get_sample(idx)

            gt = sample["gt"]
            gt_res = sample["gt_res"]
            lms = sample["lms"]
            lms_res_for_cond = sample["lms_res_for_cond"]
            pan_res = sample["pan_res"]

            cond5_res = np.concatenate([lms_res_for_cond, pan_res], axis=0)
            cond_t = (
                torch.from_numpy(cond5_res)
                .unsqueeze(0)
                .to(self.device, dtype=self.weight_dtype, non_blocking=True)
            )

            with self._autocast_context():
                if self.empty_prompt_embeds is None:
                    raise RuntimeError("empty_prompt_embeds has not been initialized.")

                output = self.pipe(
                    prompt_embeds=self.empty_prompt_embeds,
                    negative_prompt_embeds=None,
                    image=cond_t,
                    num_inference_steps=int(self.args.test_num_inference_steps),
                    generator=self.gen,
                    controlnet_conditioning_scale=float(self.args.controlnet_conditioning_scale),
                    controlnet_conditioning_scale_spa=float(self.args.controlnet_conditioning_scale_spa),
                    controlnet_conditioning_scale_spe=float(self.args.controlnet_conditioning_scale_spe),
                    guidance_scale=float(self.args.guidance_scale),
                    output_type="pt",
                )

            gen01 = output.images

            gt01 = None
            if gt is not None and gt_res is not None:
                gt01 = torch.from_numpy(gt_res).unsqueeze(0).to(
                    device=self.device,
                    dtype=torch.float32,
                )
                if gen01.shape[-2:] != gt01.shape[-2:]:
                    gen01 = F.interpolate(
                        gen01,
                        size=gt01.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            gen_img = to_pil_image(gen01[0, : min(3, gen01.shape[1])].clamp(0, 1).cpu())
            yield gen_img, {"status": "Fusion completed. Computing evaluation metrics..."}

            if gt01 is not None:
                ref_metrics, gen_hwc = self._compute_reference_metrics(gen01, gt01)
                metrics = dict(ref_metrics)
            else:
                gen_hwc = _to_hwc01(gen01)
                metrics = {}

            metrics.update(self._compute_no_reference_metrics(lms, pan_res, gen_hwc))
            yield gen_img, metrics

    def infer_one(self, idx: int):
        last_img, last_metrics = None, {}
        for image, metrics in self.infer_one_stream(idx):
            last_img, last_metrics = image, metrics

        if last_img is None:
            raise RuntimeError("infer_one_stream produced no output.")

        return last_img, last_metrics


_runner: DualBranchFusionDemoRunner | None = None


def _resolve_base_model_path(base_model_name: str, base_model_upload: Any) -> str | None:
    if base_model_name in BASE_MODEL_CHOICES:
        return _resolve_path(BASE_MODEL_CHOICES[base_model_name])

    if base_model_name == CUSTOM_BASE_KEY:
        upload_path = _unwrap_uploaded_path(base_model_upload)
        if upload_path is None:
            print("[ERROR] Custom upload was selected but no file was provided.")
            return None
        return _prepare_custom_base_model(upload_path)

    print(f"[ERROR] Unknown base model option: {base_model_name}")
    return None


def _merge_dataset_preset(
    preset_name: str,
    vae_path: str,
    dual_branch_path: str,
    test_h5_path: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    preset = DATASET_PRESETS.get(preset_name, {})
    for key, value in preset.items():
        merged[key] = _resolve_path(value)

    if vae_path.strip():
        merged["vae_path"] = _resolve_path(vae_path.strip())
    if dual_branch_path.strip():
        merged["dual_branch_model_name_or_path"] = _resolve_path(dual_branch_path.strip())
    if test_h5_path.strip():
        merged["test_h5_path"] = _resolve_path(test_h5_path.strip())

    return merged


def ui_init(
    config_path,
    base_model_name,
    base_model_upload,
    dataset_preset,
    vae_path,
    dual_branch_path,
    test_h5_path,
    eval_mode,
    range_clip_max,
    resolution,
    num_steps,
    ctrl_scale,
    ctrl_scale_spa,
    ctrl_scale_spe,
    mixed_precision,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    global _runner

    if _runner is not None:
        _runner.close()
        _runner = None

    progress(0.05, desc="Resolving base model path...")
    base_model_path = _resolve_base_model_path(base_model_name, base_model_upload)
    if not base_model_path or not Path(base_model_path).exists():
        print(f"[ERROR] Invalid base model path: {base_model_path}")
        return gr.update(), gr.update(visible=False), gr.update(visible=False)

    overrides: dict[str, Any] = {
        "pretrained_model_name_or_path": base_model_path,
        "eval_mode": eval_mode,
        "resolution": int(resolution),
        "test_num_inference_steps": int(num_steps),
        "controlnet_conditioning_scale": float(ctrl_scale),
        "controlnet_conditioning_scale_spa": float(ctrl_scale_spa),
        "controlnet_conditioning_scale_spe": float(ctrl_scale_spe),
        "mixed_precision": mixed_precision,
    }

    if range_clip_max is not None:
        overrides["range_clip_max"] = float(range_clip_max)
    if seed is not None:
        overrides["seed"] = int(seed)

    progress(0.25, desc="Applying preset and manual paths...")
    overrides.update(
        _merge_dataset_preset(
            dataset_preset,
            vae_path,
            dual_branch_path,
            test_h5_path,
        )
    )

    progress(0.60, desc="Loading models and dataset...")
    try:
        _runner = DualBranchFusionDemoRunner(str(config_path), overrides)
    except Exception:
        traceback.print_exc()
        progress(1.0, desc="Load failed")
        return gr.update(), gr.update(visible=False), gr.update(visible=False)

    progress(1.0, desc="Ready")
    return (
        gr.update(minimum=0, maximum=_runner.N - 1, value=0, step=1, visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def ui_preview(idx: int):
    global _runner

    if _runner is None:
        return None, None, None, {"error": "Please load the model and dataset first."}

    try:
        lms_img, pan_img = _runner.preview_inputs(int(idx))
        return lms_img, pan_img, None, {}
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, {"error": repr(exc)}


def ui_generate(idx: int):
    global _runner

    if _runner is None:
        yield None, {"error": "Please load the configuration and model first."}
        return

    try:
        for gen_img, metrics in _runner.infer_one_stream(int(idx)):
            yield gen_img, metrics
    except Exception as exc:
        traceback.print_exc()
        yield None, {"error": repr(exc)}


def _on_base_model_change(choice: str):
    return gr.update(visible=(choice == CUSTOM_BASE_KEY), value=None)


def _on_dataset_preset_change(preset_name: str):
    preset = DATASET_PRESETS.get(preset_name, {})
    return (
        gr.update(value=preset.get("vae_path", "")),
        gr.update(value=preset.get("dual_branch_model_name_or_path", "")),
        gr.update(value=preset.get("test_h5_path", "")),
    )


def build_demo():
    custom_css = """
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    #img_lms, #img_pan, #img_gen {
        width: 100%;
        aspect-ratio: 1 / 1;
    }

    #img_lms img, #img_pan img, #img_gen img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    """

    with gr.Blocks(css=custom_css, title="CC-PAN Demo") as demo:
        gr.HTML(
            """
            <div class="main-header">
                <h1>CC-PAN Demo</h1>
                <p style="margin-top: 0.5rem; opacity: 0.9;">
                    Remote sensing image fusion demo based on a dual-branch diffusion model
                </p>
            </div>
            """
        )

        gr.Markdown("## Configuration and inference settings")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Base configuration")
                        config_path = gr.Textbox(
                            label="Config file path",
                            value=str(DEFAULT_CONFIG_PATH),
                            placeholder="Enter the full path to the config file",
                        )

                        gr.Markdown("### Base model")
                        base_model_name = gr.Dropdown(
                            choices=list(BASE_MODEL_CHOICES.keys()) + [CUSTOM_BASE_KEY],
                            value="stable-diffusion-v1-5",
                            label="Base model",
                        )
                        base_model_upload = gr.File(
                            label="Upload custom base model",
                            type="filepath",
                            visible=False,
                            interactive=True,
                        )

                        base_model_name.change(
                            _on_base_model_change,
                            inputs=[base_model_name],
                            outputs=[base_model_upload],
                        )

                        gr.Markdown("### Preset and paths")
                        dataset_preset = gr.Dropdown(
                            choices=list(DATASET_PRESETS.keys()),
                            value="Use config values",
                            label="Dataset preset",
                        )

                        vae_path = gr.Textbox(
                            label="VAE path",
                            value="",
                            placeholder="Optional. Leave empty to use config or preset.",
                        )
                        dual_branch_path = gr.Textbox(
                            label="Dual-branch model path",
                            value="",
                            placeholder="Optional. Leave empty to use config or preset.",
                        )
                        test_h5_path = gr.Textbox(
                            label="Test H5 path",
                            value="",
                            placeholder="Optional. Leave empty to use config or preset.",
                        )

                        dataset_preset.change(
                            _on_dataset_preset_change,
                            inputs=[dataset_preset],
                            outputs=[vae_path, dual_branch_path, test_h5_path],
                        )

                        load_btn = gr.Button("Load model and dataset", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Inference settings")

                        with gr.Row():
                            eval_mode = gr.Radio(
                                choices=["reduced", "full"],
                                value="reduced",
                                label="Evaluation mode",
                            )
                            resolution = gr.Radio(
                                choices=[256, 512, 1024, 2048],
                                value=1024,
                                label="Output resolution",
                            )

                        with gr.Row():
                            range_clip_max = gr.Number(
                                value=2047.0,
                                label="Clip max value",
                            )
                            seed = gr.Number(
                                value=2025,
                                label="Random seed",
                                precision=0,
                            )
                            num_steps = gr.Slider(
                                minimum=20,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Inference steps",
                            )

                        with gr.Row():
                            mixed_precision = gr.Radio(
                                choices=["none", "fp16", "bf16"],
                                value="fp16",
                                label="Mixed precision",
                            )
                            ctrl_scale = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Global control scale",
                            )

                        with gr.Row():
                            ctrl_scale_spa = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Spatial branch scale",
                            )
                            ctrl_scale_spe = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Spectral branch scale",
                            )

                        sample_header = gr.Markdown("### Sample selection", visible=False)

                        with gr.Row():
                            idx_slider = gr.Slider(
                                label="Sample index (idx)",
                                minimum=0,
                                maximum=0,
                                value=0,
                                step=1,
                                visible=False,
                                scale=3,
                            )
                            run_btn = gr.Button(
                                "Run fusion and evaluation",
                                variant="primary",
                                visible=False,
                                scale=1,
                            )

        gr.Markdown("## Inference results")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    out_lms = gr.Image(
                        label="LRMS input",
                        type="pil",
                        elem_id="img_lms",
                    )
                    out_pan = gr.Image(
                        label="PAN input",
                        type="pil",
                        elem_id="img_pan",
                    )
                    out_gen = gr.Image(
                        label="HRMS output",
                        type="pil",
                        elem_id="img_gen",
                    )

                gr.Markdown(
                    """
                    **Image meanings**
                    - Left: LRMS input
                    - Middle: PAN input
                    - Right: fused HRMS output
                    """
                )

        with gr.Row():
            with gr.Column(scale=2):
                out_metrics = gr.JSON(
                    label="Evaluation metrics",
                    show_label=True,
                )

                gr.Markdown(
                    """
                    **Metric notes**
                    - **PSNR**: higher is better
                    - **SSIM**: higher is better
                    - **SAM**: lower is better
                    - **Q4 / ERGAS / SCC / CC**: common reference metrics
                    - **HQNR**: no-reference quality metric
                    """
                )

        load_btn.click(
            ui_init,
            inputs=[
                config_path,
                base_model_name,
                base_model_upload,
                dataset_preset,
                vae_path,
                dual_branch_path,
                test_h5_path,
                eval_mode,
                range_clip_max,
                resolution,
                num_steps,
                ctrl_scale,
                ctrl_scale_spa,
                ctrl_scale_spe,
                mixed_precision,
                seed,
            ],
            outputs=[idx_slider, run_btn, sample_header],
        )

        idx_slider.change(
            ui_preview,
            inputs=[idx_slider],
            outputs=[out_lms, out_pan, out_gen, out_metrics],
        )

        run_btn.click(
            ui_generate,
            inputs=[idx_slider],
            outputs=[out_gen, out_metrics],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
                <p>
                    Tip: the first model load may take some time. A GPU can significantly improve inference speed.
                </p>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
