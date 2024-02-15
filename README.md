# Light Schrödinger Bridge

This repository contains the `PyTorch` code to reproduce the experiments from work **Light Schrödinger Bridge** (LightSB paper on [arxiv](https://arxiv.org/abs/2310.01174)) by  [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=en&oi=ao) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

**An example:** Unpaired *Male* -> *Female* translation by our LightSB solver applied in the latent space of ALAE for 1024x1024 FFHQ images. *Our LightSB converges on 4 cpu cores in less than 1 minute.*

<p align="center"><img src="teaser/teaser.png" width="800" /></p>

## Repository structure:
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). Auxilary source code is moved to `.py` modules (`src/`). 

Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

```notebooks/Toy_experiments.ipynb``` - Toy experiments.

```ALAE``` - Code for the ALAE model.

```src``` - LightSB implementation and axuliary code for plotting.

```notebooks/LightSB_swiss_roll.ipynb``` - Code for swiss roll experiments.

```notebooks/swiss_roll_plot.ipynb``` - Code for plotting the reported image for swiss roll.

```notebooks/LightSB_EOT_benchmark.ipynb``` - Code for benchmark experiments.

```notebooks/LightSB_single_cell.ipynb``` - Code for single cell experiments.

```notebooks/LightSB_alae.ipynb``` - Code for image experiments with ALAE.

## Citation
```
@inproceedings{
korotin2024light,
title={Light Schr\"odinger Bridge},
author={Alexander Korotin and Nikita Gushchin and Evgeny Burnaev},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=WhZoCLRWYJ}
}
```
