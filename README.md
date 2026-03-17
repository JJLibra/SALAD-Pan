<h2 align="center">SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening</h2>

<p align="center">
  <a href="https://scholar.google.com/citations?hl=en&user=Jo_8lVcAAAAJ">Junjie Li</a> ·
  <a href="https://github.com/ocy1">Congyang Ou</a> ·
  <a href="https://scholar.google.com/citations?hl=en&user=m3gPwCoAAAAJ">Haokui Zhang</a> ·
  <a href="https://scholar.google.com/citations?hl=en&user=NW8rUFkAAAAJ">Guoting Wei</a> ·
  <a href="https://ieeexplore.ieee.org/author/37086409411">Shengqin Jiang</a> ·
  <a href="https://teacher.nwpu.edu.cn/2005000096.html">Ying Li</a> ·
  <a href="https://scholar.google.com/citations?hl=en&user=Ljk2BvIAAAAJ">Chunhua Shen</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.04473">
    <img src="https://img.shields.io/badge/arXiv-2602.04473-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/xxfer/SALAD-Pan">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a>
  <a href="https://salad-pan.github.io">
    <img src="https://img.shields.io/badge/Project-Website-orange" alt="Project Website">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10.12-blue" alt="Python Version">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.10-yellow" alt="PyTorch Version">
  </a>
</p>

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig1.pdf">
    <img src="https://salad-pan.github.io/assets/fig1.webp" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>Given a PAN–LRMS image pair, SALAD-Pan fine-tunes a pre-trained diffusion model to generate a HRMS.</em>
</p>

## News
<!-- ### 🚨 Announcing [](): A CVPR competition for AI-based xxxxxx! Submissions due xxx x. Don't miss out! 🤩  -->
- [02/01/2026] Code will be released soon !
<!-- - [04/30/2026] Pre-trained SALAD-Pan models are available on [Hugging Face Library](https://huggingface.co/xxfer/SALAD-Pan)! -->
<!-- - [05/01/2026] Code released! -->

## Contents

- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [Shoutouts](#shoutouts)

## Setup

### Requirements

```shell
git clone https://github.com/JJLibra/SALAD-Pan.git
cd SALAD-Pan

conda create -n saladpan python=3.10 -y
conda activate saladpan
pip install -r requirements.txt

# This project depends on a modified local version of `diffusers` under `./diffusers`.
cd diffusers
pip install -e .
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True`.

### Weights

We provide **two-stage checkpoints**:

- **Stage I (Band-VAE)**: `checkpoints/vae.safetensors` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))
- **Stage II (Latent Diffusion)**: runs **on top of Stable Diffusion** in the Band-VAE latent space.  
  - **Stable Diffusion base**: download from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5))  
  - **Adapters**: `checkpoints/adapters.pth` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))

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

**🚨We strongly recommend that you visit this [website](https://salad-pan.github.io/) for a better reading experience.**

<p><b>Table 1.</b> Quantitative results on the WorldView-3 (WV3) dataset. Best and second-best results are in <b>bold</b> and <u>underlined</u>.</p>

<div style="overflow-x:auto; width:100%;">
  <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%; min-width:980px; font-size:13px; white-space:nowrap;">
    <thead>
      <tr>
        <th style="text-align:left;">Models</th>
        <th style="text-align:right;">Pub/Year</th>
        <th style="text-align:right;">Q<sub>8</sub> ↑</th>
        <th style="text-align:right;">SAM ↓</th>
        <th style="text-align:right;">ERGAS ↓</th>
        <th style="text-align:right;">SCC ↑</th>
        <th style="text-align:right;">D<sub>λ</sub> ↓</th>
        <th style="text-align:right;">D<sub>s</sub> ↓</th>
        <th style="text-align:right;">HQNR ↑</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>PaNNet</td><td style="text-align:right;">ICCV’17</td><td style="text-align:right;">0.891±0.045</td><td style="text-align:right;">3.613±0.787</td><td style="text-align:right;">2.664±0.347</td><td style="text-align:right;">0.943±0.018</td><td style="text-align:right;">0.017±0.008</td><td style="text-align:right;">0.047±0.014</td><td style="text-align:right;">0.937±0.015</td></tr>
      <tr><td>FusionNet</td><td style="text-align:right;">TGRS’20</td><td style="text-align:right;">0.904±0.092</td><td style="text-align:right;">3.324±0.411</td><td style="text-align:right;">2.465±0.603</td><td style="text-align:right;">0.958±0.023</td><td style="text-align:right;">0.024±0.011</td><td style="text-align:right;">0.036±0.016</td><td style="text-align:right;">0.940±0.019</td></tr>
      <tr><td>LAGConv</td><td style="text-align:right;">AAAI’22</td><td style="text-align:right;">0.910±0.114</td><td style="text-align:right;">3.104±1.119</td><td style="text-align:right;">2.300±0.911</td><td style="text-align:right;">0.980±0.043</td><td style="text-align:right;">0.036±0.009</td><td style="text-align:right;">0.032±0.016</td><td style="text-align:right;">0.934±0.011</td></tr>
      <tr><td>BiMPAN</td><td style="text-align:right;">ACMM’23</td><td style="text-align:right;">0.915±0.087</td><td style="text-align:right;">2.984±0.601</td><td style="text-align:right;">2.257±0.552</td><td style="text-align:right;">0.984±0.005</td><td style="text-align:right;">0.017±0.019</td><td style="text-align:right;">0.035±0.015</td><td style="text-align:right;">0.949±0.026</td></tr>
      <tr><td>ARConv</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;">0.916±0.083</td><td style="text-align:right;">2.858±0.590</td><td style="text-align:right;">2.117±0.528</td><td style="text-align:right;">0.989±0.014</td><td style="text-align:right;">0.014±0.006</td><td style="text-align:right;">0.030±0.007</td><td style="text-align:right;">0.958±0.010</td></tr>
      <tr><td>WFANET</td><td style="text-align:right;">AAAI’25</td><td style="text-align:right;">0.917±0.088</td><td style="text-align:right;">2.855±0.618</td><td style="text-align:right;">2.095±0.422</td><td style="text-align:right;"><u>0.989±0.011</u></td><td style="text-align:right;">0.012±0.007</td><td style="text-align:right;">0.031±0.009</td><td style="text-align:right;">0.957±0.010</td></tr>
      <tr><td>PanDiff</td><td style="text-align:right;">TGRS’23</td><td style="text-align:right;">0.898±0.090</td><td style="text-align:right;">3.297±0.235</td><td style="text-align:right;">2.467±0.166</td><td style="text-align:right;">0.980±0.019</td><td style="text-align:right;">0.027±0.108</td><td style="text-align:right;">0.054±0.047</td><td style="text-align:right;">0.920±0.077</td></tr>
      <tr><td>SSDiff</td><td style="text-align:right;">NeurIPS’24</td><td style="text-align:right;">0.915±0.086</td><td style="text-align:right;">2.843±0.529</td><td style="text-align:right;">2.106±0.416</td><td style="text-align:right;">0.986±0.004</td><td style="text-align:right;">0.013±0.005</td><td style="text-align:right;">0.031±0.003</td><td style="text-align:right;">0.956±0.016</td></tr>
      <tr><td>SGDiff</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;"><u>0.921±0.082</u></td><td style="text-align:right;"><u>2.771±0.511</u></td><td style="text-align:right;"><u>2.044±0.449</u></td><td style="text-align:right;">0.987±0.009</td><td style="text-align:right;"><u>0.012±0.005</u></td><td style="text-align:right;"><u>0.027±0.003</u></td><td style="text-align:right;"><u>0.960±0.006</u></td></tr>
      <tr><td><b>SALAD&#8209;Pan</b></td><td style="text-align:right;"><b>Ours</b></td><td style="text-align:right;"><b>0.924±0.064</b></td><td style="text-align:right;"><b>2.689±0.135</b></td><td style="text-align:right;"><b>1.839±0.211</b></td><td style="text-align:right;"><b>0.989±0.007</b></td><td style="text-align:right;"><b>0.010±0.008</b></td><td style="text-align:right;"><b>0.021±0.004</b></td><td style="text-align:right;"><b>0.965±0.007</b></td></tr>
    </tbody>
  </table>
</div>

<br/>

<p><b>Table 2.</b> Quantitative results on the QuickBird (QB) dataset. Best and second-best results are in <b>bold</b> and <u>underlined</u>.</p>

<div style="overflow-x:auto; width:100%;">
  <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%; min-width:980px; font-size:13px; white-space:nowrap;">
    <thead>
      <tr>
        <th style="text-align:left;">Models</th>
        <th style="text-align:right;">Pub/Year</th>
        <th style="text-align:right;">Q<sub>4</sub> ↑</th>
        <th style="text-align:right;">SAM ↓</th>
        <th style="text-align:right;">ERGAS ↓</th>
        <th style="text-align:right;">SCC ↑</th>
        <th style="text-align:right;">D<sub>λ</sub> ↓</th>
        <th style="text-align:right;">D<sub>s</sub> ↓</th>
        <th style="text-align:right;">HQNR ↑</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>PaNNet</td><td style="text-align:right;">ICCV’17</td><td style="text-align:right;">0.885±0.118</td><td style="text-align:right;">5.791±0.995</td><td style="text-align:right;">5.863±0.413</td><td style="text-align:right;">0.948±0.021</td><td style="text-align:right;">0.059±0.017</td><td style="text-align:right;">0.061±0.010</td><td style="text-align:right;">0.883±0.025</td></tr>
      <tr><td>FusionNet</td><td style="text-align:right;">TGRS’20</td><td style="text-align:right;">0.925±0.087</td><td style="text-align:right;">4.923±0.812</td><td style="text-align:right;">4.159±0.351</td><td style="text-align:right;">0.956±0.018</td><td style="text-align:right;">0.059±0.019</td><td style="text-align:right;">0.052±0.009</td><td style="text-align:right;">0.892±0.022</td></tr>
      <tr><td>LAGConv</td><td style="text-align:right;">AAAI’22</td><td style="text-align:right;">0.916±0.130</td><td style="text-align:right;">4.370±0.720</td><td style="text-align:right;">3.740±0.290</td><td style="text-align:right;">0.959±0.047</td><td style="text-align:right;">0.085±0.024</td><td style="text-align:right;">0.068±0.014</td><td style="text-align:right;">0.853±0.018</td></tr>
      <tr><td>BiMPAN</td><td style="text-align:right;">ACMM’23</td><td style="text-align:right;">0.931±0.091</td><td style="text-align:right;">4.586±0.821</td><td style="text-align:right;">3.840±0.319</td><td style="text-align:right;">0.980±0.008</td><td style="text-align:right;">0.026±0.020</td><td style="text-align:right;">0.040±0.013</td><td style="text-align:right;">0.935±0.030</td></tr>
      <tr><td>ARConv</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;">0.936±0.088</td><td style="text-align:right;">4.453±0.499</td><td style="text-align:right;">3.649±0.401</td><td style="text-align:right;"><b>0.987±0.009</b></td><td style="text-align:right;"><u>0.019±0.014</u></td><td style="text-align:right;">0.034±0.017</td><td style="text-align:right;">0.948±0.042</td></tr>
      <tr><td>WFANET</td><td style="text-align:right;">AAAI’25</td><td style="text-align:right;">0.935±0.092</td><td style="text-align:right;">4.490±0.582</td><td style="text-align:right;">3.604±0.337</td><td style="text-align:right;"><u>0.986±0.008</u></td><td style="text-align:right;">0.019±0.016</td><td style="text-align:right;"><u>0.033±0.019</u></td><td style="text-align:right;"><u>0.948±0.037</u></td></tr>
      <tr><td>PanDiff</td><td style="text-align:right;">TGRS’23</td><td style="text-align:right;">0.934±0.095</td><td style="text-align:right;">4.575±0.255</td><td style="text-align:right;">3.742±0.353</td><td style="text-align:right;">0.980±0.007</td><td style="text-align:right;">0.058±0.015</td><td style="text-align:right;">0.064±0.020</td><td style="text-align:right;">0.881±0.075</td></tr>
      <tr><td>SSDiff</td><td style="text-align:right;">NeurIPS’24</td><td style="text-align:right;">0.934±0.094</td><td style="text-align:right;">4.464±0.747</td><td style="text-align:right;">3.632±0.275</td><td style="text-align:right;">0.982±0.008</td><td style="text-align:right;">0.031±0.011</td><td style="text-align:right;">0.036±0.013</td><td style="text-align:right;">0.934±0.021</td></tr>
      <tr><td>SGDiff</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;"><u>0.938±0.087</u></td><td style="text-align:right;"><u>4.353±0.741</u></td><td style="text-align:right;"><u>3.578±0.290</u></td><td style="text-align:right;">0.983±0.007</td><td style="text-align:right;">0.023±0.013</td><td style="text-align:right;">0.043±0.012</td><td style="text-align:right;">0.934±0.011</td></tr>
      <tr><td><b>SALAD&#8209;Pan</b></td><td style="text-align:right;"><b>Ours</b></td><td style="text-align:right;"><b>0.939±0.088</b></td><td style="text-align:right;"><b>4.198±0.526</b></td><td style="text-align:right;"><b>3.251±0.288</b></td><td style="text-align:right;">0.984±0.009</td><td style="text-align:right;"><b>0.017±0.011</b></td><td style="text-align:right;"><b>0.026±0.009</b></td><td style="text-align:right;"><b>0.957±0.010</b></td></tr>
    </tbody>
  </table>
</div>

<br/>

<p><b>Table 3.</b> Quantitative results on the GaoFen-2 (GF2) dataset. Best and second-best results are in <b>bold</b> and <u>underlined</u>.</p>

<div style="overflow-x:auto; width:100%;">
  <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%; min-width:980px; font-size:13px; white-space:nowrap;">
    <thead>
      <tr>
        <th style="text-align:left;">Models</th>
        <th style="text-align:right;">Pub/Year</th>
        <th style="text-align:right;">Q<sub>4</sub> ↑</th>
        <th style="text-align:right;">SAM ↓</th>
        <th style="text-align:right;">ERGAS ↓</th>
        <th style="text-align:right;">SCC ↑</th>
        <th style="text-align:right;">D<sub>λ</sub> ↓</th>
        <th style="text-align:right;">D<sub>s</sub> ↓</th>
        <th style="text-align:right;">HQNR ↑</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>PaNNet</td><td style="text-align:right;">ICCV’17</td><td style="text-align:right;">0.967±0.013</td><td style="text-align:right;">0.997±0.022</td><td style="text-align:right;">0.919±0.039</td><td style="text-align:right;">0.973±0.011</td><td style="text-align:right;">0.017±0.012</td><td style="text-align:right;">0.047±0.012</td><td style="text-align:right;">0.937±0.023</td></tr>
      <tr><td>FusionNet</td><td style="text-align:right;">TGRS’20</td><td style="text-align:right;">0.964±0.014</td><td style="text-align:right;">0.974±0.035</td><td style="text-align:right;">0.988±0.072</td><td style="text-align:right;">0.971±0.012</td><td style="text-align:right;">0.040±0.013</td><td style="text-align:right;">0.101±0.014</td><td style="text-align:right;">0.863±0.018</td></tr>
      <tr><td>LAGConv</td><td style="text-align:right;">AAAI’22</td><td style="text-align:right;">0.970±0.011</td><td style="text-align:right;">1.080±0.023</td><td style="text-align:right;">0.910±0.045</td><td style="text-align:right;">0.977±0.006</td><td style="text-align:right;">0.033±0.013</td><td style="text-align:right;">0.079±0.013</td><td style="text-align:right;">0.891±0.021</td></tr>
      <tr><td>BiMPAN</td><td style="text-align:right;">ACMM’23</td><td style="text-align:right;">0.965±0.020</td><td style="text-align:right;">0.902±0.066</td><td style="text-align:right;">0.881±0.058</td><td style="text-align:right;">0.972±0.018</td><td style="text-align:right;">0.032±0.015</td><td style="text-align:right;">0.051±0.014</td><td style="text-align:right;">0.918±0.019</td></tr>
      <tr><td>ARConv</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;">0.982±0.013</td><td style="text-align:right;">0.710±0.149</td><td style="text-align:right;">0.645±0.127</td><td style="text-align:right;"><u>0.994±0.005</u></td><td style="text-align:right;">0.007±0.005</td><td style="text-align:right;">0.029±0.019</td><td style="text-align:right;">0.963±0.018</td></tr>
      <tr><td>WFANET</td><td style="text-align:right;">AAAI’25</td><td style="text-align:right;">0.981±0.007</td><td style="text-align:right;">0.751±0.082</td><td style="text-align:right;">0.657±0.074</td><td style="text-align:right;"><b>0.994±0.002</b></td><td style="text-align:right;"><b>0.003±0.003</b></td><td style="text-align:right;">0.032±0.021</td><td style="text-align:right;"><u>0.964±0.020</u></td></tr>
      <tr><td>PanDiff</td><td style="text-align:right;">TGRS’23</td><td style="text-align:right;">0.979±0.011</td><td style="text-align:right;">0.888±0.037</td><td style="text-align:right;">0.746±0.031</td><td style="text-align:right;">0.988±0.003</td><td style="text-align:right;">0.027±0.011</td><td style="text-align:right;">0.073±0.013</td><td style="text-align:right;">0.903±0.025</td></tr>
      <tr><td>SSDiff</td><td style="text-align:right;">NeurIPS’24</td><td style="text-align:right;"><b>0.983±0.007</b></td><td style="text-align:right;"><u>0.670±0.124</u></td><td style="text-align:right;"><u>0.604±0.108</u></td><td style="text-align:right;">0.991±0.006</td><td style="text-align:right;">0.016±0.009</td><td style="text-align:right;">0.027±0.027</td><td style="text-align:right;">0.957±0.010</td></tr>
      <tr><td>SGDiff</td><td style="text-align:right;">CVPR’25</td><td style="text-align:right;">0.980±0.011</td><td style="text-align:right;">0.708±0.119</td><td style="text-align:right;">0.668±0.094</td><td style="text-align:right;">0.989±0.005</td><td style="text-align:right;">0.020±0.013</td><td style="text-align:right;"><u>0.024±0.022</u></td><td style="text-align:right;">0.959±0.011</td></tr>
      <tr><td><b>SALAD&#8209;Pan</b></td><td style="text-align:right;"><b>Ours</b></td><td style="text-align:right;"><u>0.982±0.010</u></td><td style="text-align:right;"><b>0.667±0.051</b></td><td style="text-align:right;"><b>0.592±0.088</b></td><td style="text-align:right;">0.991±0.003</td><td style="text-align:right;"><u>0.005±0.002</u></td><td style="text-align:right;"><b>0.022±0.014</b></td><td style="text-align:right;"><b>0.973±0.010</b></td></tr>
    </tbody>
  </table>
</div>

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig3.pdf">
    <!-- <img src="https://salad-pan.github.io/assets/fig3-1.png" alt="Reduced Resolution" width="100%" /> -->
    <img src="https://salad-pan.github.io/assets/fig3.webp" alt="Reduced Resolution" width="100%" />
  </a>
  <br>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at reduced resolution.</em>
  <a href="https://salad-pan.github.io/assets/fig4.pdf">
    <!-- <img src="https://salad-pan.github.io/assets/fig4-1.png" alt="Full Resolution" width="100%" /> -->
    <img src="https://salad-pan.github.io/assets/fig4.webp" alt="Full Resolution" width="100%" />
  </a>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at full resolution.</em>
</p>

### Efficiency comparison (RR, QB)

| Diffusion-based Methods |           SAM ↓ |         ERGAS ↓ |  NFE | Latency (s) ↓ |
| ----------------------- | --------------: | --------------: | ---: | ------------: |
| PanDiff                 |     4.575±0.255 |     3.742±0.353 | 1000 |   356.63±1.98 |
| SSDiff                  |     4.464±0.747 |     3.632±0.275 |   10 |    10.10±0.21 |
| SGDiff                  |     4.353±0.741 |     3.578±0.290 |   50 |     6.64±0.09 |
| **SALAD-Pan**           | **4.198±0.526** | **3.251±0.288** |   20 | **3.36±0.07** |

> Latency is reported as mean ± std over 10 runs (warmup=3), batch size=1, evaluated on the QB dataset under the reduced-resolution (RR) protocol, on an RTX 4090 GPU.

## Citation

If you make use of our work, please cite our paper.

```bibtex
@article{li2026saladpan,
  title={SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening},
  author={Junjie Li and Congyang Ou and Haokui Zhang and Guoting Wei and Shengqin Jiang and Ying Li and Chunhua Shen},
  journal={arXiv preprint arXiv:2602.04473},
  year={2026}
}
```

## Shoutouts

- Built with [🤗 Diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing !
- The interactive demo is powered by [🤗 Gradio](https://github.com/gradio-app/gradio). Thanks for open-sourcing !
