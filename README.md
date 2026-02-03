# SALAD-Pan

This repository is the official implementation of [SALAD-Pan]().

**[SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening]()**
<br/>
[Junjie Li](), 
[Congyang Ou](), 
[Haokui Zhang](), 
[Guoting Wei](), 
[Shengqin Jiang](), 
[Ying Li](),
[Chunhua Shen]()
<br/>

<!-- [![Project Website](https://img.shields.io/badge/Project-Website-orange)]() -->
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/xxfer/SALAD-Pan)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() -->

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig1.pdf">
    <img src="https://salad-pan.github.io/assets/fig1-1.png" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>Given a pan-ms pair as input, our method, SALAD-Pan, fine-tunes a pre-trained text-to-image diffusion model for pansharpening.</em>
</p>

## News
<!-- ### 🚨 Announcing [](): A CVPR competition for AI-based xxxxxx! Submissions due xxx x. Don't miss out! 🤩  -->
- [02/03/2026] Code will be released soon!
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

We release **two-stage weights**:

- **Stage I (Band-VAE)**: a single-band VAE pretrained to build a compact latent space.  
  - **[VAE]**: `models/vae.safetensors`. You can download it from [Hugging face](https://huggingface.co/xxfer/SALAD-Pan).

- **Stage II (Latent Diffusion)**: a latent conditional diffusion model trained **on top of Stable Diffusion**, operating in the Band-VAE latent space with spatial-spectral conditioning.  
  - **[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)).  
  - **[Adapters]**: `models/adapters.pth`. You can download it from [Hugging face](https://huggingface.co/xxfer/SALAD-Pan).

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

## Citation

If you make use of our work, please cite our paper.

```bibtex
```

## Shoutouts

- Built with [🤗 Diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!
- The interactive demo is powered by [🤗 Gradio](https://github.com/gradio-app/gradio). Thanks for open-sourcing!
