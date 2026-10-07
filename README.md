# Sketch-Text-Img2Img: Reproducibility Repository

This repository contains the reproducibility workflow and code for the paper:

**Sketch–Text–Img2Img: A Pilot Quantitative Evaluation of a Multimodal Human–AI Collaborative Workflow for Early-Stage Visual Concept Design**

## 1. Overview

The workflow compares two conditions:

- **Text-Only**: iterative text prompting with SDXL.
- **Sketch-Loop**: hand sketch → CLIP Interrogator → img2img with ControlNet Canny.

The repository includes:

- Synthetic data generation for pipeline testing.
- Main experiment, ablation, automated sketch capture, multi-task validation, CLIP Interrogator validation, and professional rating templates.
- Statistical analysis: bootstrap 95% CI, Cohen's d, FDR correction.
- Figure generation code for Fig. 1–6.

**Important**: The `data/` folder contains synthetic example data. Replace it with real experimental data before publication.

## 2. Environment

- Python 3.10.11
- PyTorch 2.1.0+cu121
- Diffusers 0.25.0
- ControlNet: `diffusers/controlnet-canny-sdxl-1.0`
- CLIP Interrogator: ViT-L-14, classic mode
- GPU: NVIDIA RTX 4070 Super 12GB or equivalent

Install dependencies:

```bash
pip install -r requirements.txt
