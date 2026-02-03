# SALAD-Pan

This repository is the official implementation of [SALAD-Pan](https://arxiv.org/abs/2212.11565).

**[SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening](https://arxiv.org/abs/2212.11565)**
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
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2212.11565)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]()
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() -->

<!-- <p align="center">
<img src="https://tuneavideo.github.io/assets/teaser.gif" width="1080px"/>  
<br>
<em>Given a pan-ms pair as input, our method, SALAD-Pan, fine-tunes a pre-trained text-to-image diffusion model for pansharpening.</em>
</p> -->

<!-- ## News
### 🚨 Announcing [LOVEU-TGVE](https://sites.google.com/view/loveucvpr23/track4): A CVPR competition for AI-based video editing! Submissions due Jun 5. Don't miss out! 🤩 
- [02/22/2023] Improved consistency using DDIM inversion.
- [02/08/2023] [Colab demo](https://colab.research.google.com/github/showlab/Tune-A-Video/blob/main/notebooks/Tune-A-Video.ipynb) released!
- [02/03/2023] Pre-trained Tune-A-Video models are available on [Hugging Face Library](https://huggingface.co/Tune-A-Video-library)!
- [01/28/2023] New Feature: tune a video on personalized [DreamBooth](https://dreambooth.github.io/) models.
- [01/28/2023] Code released! -->

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)).

<!-- **[VAE]** [VAE]() xxx ...... -->

## Usage

### Training

, run this command:

```bash
accelerate launch train_vae.py --config configs/train_vae.yaml
```

, run this command:

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

<!-- <table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/man-skiing.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/spiderman-beach.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/wonder-woman.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-skiing/pink-sunset.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is skiing"</td>
  <td width=25% style="text-align:center;">"Spider Man is skiing on the beach, cartoon style”</td>
  <td width=25% style="text-align:center;">"Wonder Woman, wearing a cowboy hat, is skiing"</td>
  <td width=25% style="text-align:center;">"A man, wearing pink clothes, is skiing at sunset"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/rabbit-watermelon.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/rabbit.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/cat.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/rabbit-watermelon/puppy.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A rabbit is eating a watermelon on the table"</td>
  <td width=25% style="text-align:center;">"A rabbit is <del>eating a watermelon</del> on the table"</td>
  <td width=25% style="text-align:center;">"A cat with sunglasses is eating a watermelon on the beach"</td>
  <td width=25% style="text-align:center;">"A puppy is eating a cheeseburger on the table, comic style"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/car-turn.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/porsche-beach.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/car-cartoon.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/car-snow.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A jeep car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A Porsche car is moving on the beach"</td>
  <td width=25% style="text-align:center;">"A car is moving on the road, cartoon style"</td>
  <td width=25% style="text-align:center;">"A car is moving on the snow"</td>
</tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/man-basketball.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/bond.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/astronaut.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/man-basketball/lego.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is dribbling a basketball"</td>
  <td width=25% style="text-align:center;">"James Bond is dribbling a basketball on the beach"</td>
  <td width=25% style="text-align:center;">"An astronaut is dribbling a basketball, cartoon style"</td>
  <td width=25% style="text-align:center;">"A lego man in a black suit is dribbling a basketball"</td>
</tr>
</table> -->

<!-- ## Citation
If you make use of our work, please cite our paper.
```bibtex``` -->

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!
<!-- - Thanks [gradio demo](). -->
