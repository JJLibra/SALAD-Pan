# SALAD-Pan

This repository is the official implementation of [SALAD-Pan]().

**[SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening]()**
<br/>
[Junjie Li](https://scholar.google.com/citations?hl=en&user=Jo_8lVcAAAAJ), 
[Congyang Ou](https://github.com/ocy1), 
[Haokui Zhang](https://scholar.google.com/citations?hl=en&user=m3gPwCoAAAAJ), 
[Guoting Wei](https://scholar.google.com/citations?hl=en&user=NW8rUFkAAAAJ), 
[Shengqin Jiang](https://ieeexplore.ieee.org/author/37086409411), 
[Ying Li](), 
[Chunhua Shen](https://scholar.google.com/citations?hl=en&user=Ljk2BvIAAAAJ)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://salad-pan.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/xxfer/SALAD-Pan)

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig1.pdf">
    <img src="https://salad-pan.github.io/assets/fig1-1.png" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>Given a PAN–LRMS image pair, SALAD-Pan fine-tunes a pre-trained diffusion model to generate a HRMS.</em>
</p>

## News
<!-- ### 🚨 Announcing [](): A CVPR competition for AI-based xxxxxx! Submissions due xxx x. Don't miss out! 🤩  -->
- [02/01/2026] Code will be released soon!
<!-- - [04/30/2026] Pre-trained SALAD-Pan models are available on [Hugging Face Library](https://huggingface.co/xxfer/SALAD-Pan)! -->
<!-- - [05/01/2026] Code released! -->

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True`.

### Weights

We provide **two-stage checkpoints**:

- **Stage I (Band-VAE)**: `models/vae.safetensors` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))
- **Stage II (Latent Diffusion)**: runs **on top of Stable Diffusion** in the Band-VAE latent space.  
  - **Stable Diffusion base**: download from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5))  
  - **Adapters**: `models/adapters.pth` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))

## Usage

### Training

We train the model in **two stages**.

- **Stage I (VAE pretraining)**

```bash
accelerate launch train_vae.py --config configs/train_vae.yaml
```

- **Stage II (Diffusion + Adapter training)**

```bash
accelerate launch train_diffusion.py --config configs/train_diffusion.yaml
```

Note: Tuning usually takes `40k~50k` steps, about `1~2` days using eight RTX 4090 GPUs in fp16. 
Reduce `batch_size` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
Coming soon.
```

## Results

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig3.pdf">
    <img src="https://salad-pan.github.io/assets/fig3-1.png" alt="Reduced Resolution" width="100%" />
  </a>
  <br>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at reduced resolution.</em>
  <a href="https://salad-pan.github.io/assets/fig4.pdf">
    <img src="https://salad-pan.github.io/assets/fig4-1.png" alt="Full Resolution" width="100%" />
  </a>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at full resolution.</em>
</p>

### Inference speed

| Diffusion-based Methods | Settings | Latency (s) ↓ |
|---|---|---:|
| PanDiff   | fp16 · 256×256 | 356.63 ± 1.98 |
| SSDiff    | fp16 · 256×256 | 10.10 ± 0.21  |
| SGDiff    | fp16 · 256×256 | 6.64 ± 0.09   |
| SALAD-Pan | fp16 · 256×256 | 3.36 ± 0.07   |

> Latency is reported as mean ± std over 10 runs (warmup=3), batch size=1, on <RTX 4090 GPU>.

## Citation

If you make use of our work, please cite our paper.

```bibtex
```

## Shoutouts

- Built with [🤗 Diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing !
- The interactive demo is powered by [🤗 Gradio](https://github.com/gradio-app/gradio). Thanks for open-sourcing !
