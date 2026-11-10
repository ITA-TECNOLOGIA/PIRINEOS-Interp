# PIRINEOS-Interp (*Probing & Interpreting Reasoning In Neural Entropy with Outputs & Statistics*)

**Code & data for our xAI 2026 Late-Breaking Work paper:** *Delayed Convergence and Emergent CoT Reliance in RL-Tuned Language Models*
Presented at the Late-Breaking Work, Demos and Doctoral Consortium track colocated with the **4th World Conference on eXplainable Artificial Intelligence (xAI 2026)**, Fortaleza, Brazil, July 1–3, 2026.
We study how **reasoning** emerges in LLMs using two complementary, activation‑level methods:

1. **Tuned‑Lens probing** — to quantify how intermediate layer predictions and calibration shift across models (base vs. RL‑fine‑tuned vs. distilled).
2. **Integrated Gradients (IG)** — to compare the influence of the **prompt** versus **intermediate reasoning tokens** over the model’s final answer.

Target models and benchmark:

* **Qwen/Qwen2.5‑14B** (base),
* **GRPO‑trained Qwen2.5‑14B** (this repo’s checkpoint),
* **deepseek‑ai/DeepSeek‑R1‑Distill‑Qwen‑14B**,
* **GSM8K** for math word problems.

---

## Contents

```
.
├── src/main/python/eval_gsm8k/
│   ├── tuned_lens_evaluation.py      # Tuned‑Lens pipeline on GSM8K
│   └── ig_evaluation.py              # Integrated Gradients pipeline on GSM8K
├── notebooks/
│   ├── Qwen2_5_(14B)_GRPO.py         # GRPO training pipeline for the reasoning model
│   ├── tunedlens_gsm8k_20250610_copy20250615.py   # TL statistics & figures
│   └── ig_gsm8k_20250612_copy20250615.py          # IG statistics & figures
├── tuned-lens-ITA/                   # Tuned‑Lens checkpoints (as submodule)
├── outputs/                          # GRPO checkpoints (e.g., checkpoint-1500)
├── requirements.txt                  # (provided; see below for versions)
└── README.md
```

---

## Setup

### 1) Python & CUDA

We used **Python 3.11** and **CUDA 12.2**. A safe baseline:

```bash
conda create -n qwen-tunedlens-ig python=3.11 -y
conda activate qwen-tunedlens-ig

# PyTorch for CUDA 12.1/12.2 (adjust if your CUDA differs)
pip install torch==2.6.0 torchvision==0.21.0
```

### 2) Core libraries

We rely on:

* `transformers`, `datasets`, `accelerate`
* `tuned-lens`
* `captum`
* `vllm==0.6.6.post1` (for inference backends where applicable)
* scientific stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly==5.13.1`

Install everything:

```bash
# Project requirements
pip install -r requirements.txt
```

### 3) Submodules (Captum & Tuned‑Lens forks)

The repo tracks the required forks/weights as submodules. After cloning this repo:

```bash
git submodule update --init --recursive

# Editable installs of our submodules if needed:
cd captum-ITA && pip install -e . && cd -
cd tuned-lens-ITA && pip install -e . && cd -
```

---

## Data & Models

* **Benchmark**: GSM8K (auto‑downloaded via `datasets` on first run).
* **Models**:

  * Base: `Qwen/Qwen2.5-14B`
  * RL‑fine‑tuned (GRPO): `outputs/Qwen2_5_14B_GRPO/checkpoint-1500` (trained with `notebooks/Qwen2_5_(14B)_GRPO.py`)
  * Distilled: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

### Tuned-Lens Weights

We provide layer-wise Tuned-Lens weights for `Qwen/Qwen2.5-14B` in the followingHugging Face repo: `nq7z3kdm/Qwen2.5-14B-Lens`.

These weights were trained on the test split of The Pile. They are released for interpretability and qualitative analysis. They have to be available in the `my_lenses` folder inside Tuned-Lens, as Qwen2.5 wasn't supported by default.

---

## Reproducing Paper Experiments

Use a single visible GPU via `CUDA_VISIBLE_DEVICES`. We ran on 80GB‑class GPUs.
Logs below redirect to `logs/*.log`; feel free to remove redirection to stream to console.

### A) Tuned‑Lens (TL) on GSM8K

**Base (Qwen2.5‑14B)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=3

nohup python src/main/python/eval_gsm8k/tuned_lens_evaluation.py \
  --model Qwen/Qwen2.5-14B \
  --tuned_lens tuned-lens-ITA/my_lenses/Qwen2/Qwen25-14B \
  --max_new_tokens 1536 \
  > logs/eval_gsm8k_tunedlens_base_1536_final.log &
```

**GRPO (checkpoint‑1500)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=1

nohup python src/main/python/eval_gsm8k/tuned_lens_evaluation.py \
  --model outputs/Qwen2_5_14B_GRPO/checkpoint-1500 \
  --tuned_lens tuned-lens-ITA/my_lenses/Qwen2/Qwen25-14B \
  --max_new_tokens 2048 \
  > logs/eval_gsm8k_tunedlens_grpo_2048_final.log &
```

**Distilled (DeepSeek‑R1‑Distill‑Qwen‑14B, quantized)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=3

nohup python src/main/python/eval_gsm8k/tuned_lens_evaluation.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --tuned_lens tuned-lens-ITA/my_lenses/Qwen2/Qwen25-14B \
  --max_new_tokens 4096 \
  --quantized \
  > logs/eval_gsm8k_tunedlens_distil_4096_final.log &
```

### B) Integrated Gradients (IG) on GSM8K

**Base (Qwen2.5‑14B, quantized)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=0

nohup python src/main/python/eval_gsm8k/ig_evaluation.py \
  --model Qwen/Qwen2.5-14B \
  --max-new-tokens 1536 \
  --n-steps 25 \
  --internal-batch-size 1 \
  --quantized \
  > logs/eval_gsm8k_ig_base_1536_final.log &
```

**GRPO (checkpoint‑1500, quantized)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=1

nohup python src/main/python/eval_gsm8k/ig_evaluation.py \
  --model outputs/Qwen2_5_14B_GRPO/checkpoint-1500 \
  --max-new-tokens 2048 \
  --n-steps 25 \
  --internal-batch-size 1 \
  --quantized \
  > logs/eval_gsm8k_ig_grpo_2048_final.log &
```

**Distilled (DeepSeek‑R1‑Distill‑Qwen‑14B, quantized)**

```bash
conda activate qwen-tunedlens-ig
export HF_TOKEN=***your_hf_token***
export CUDA_VISIBLE_DEVICES=3

nohup python src/main/python/eval_gsm8k/ig_evaluation.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --max-new-tokens 4096 \
  --n-steps 15 \
  --internal-batch-size 1 \
  --quantized \
  > logs/eval_gsm8k_ig_distil_4096_final_20250612.log &
```

---

## Analysis Notebooks

Post‑processing and figures:

* `notebooks/Qwen2_5_(14B)_GRPO.py` — GRPO training pipeline for the reasoning model.
* `notebooks/tunedlens_gsm8k_20250610_copy20250615.py` — layer‑wise TL stats & plots across GSM8K.
* `notebooks/ig_gsm8k_20250612_copy20250615.py` — IG attribution stats & plots across GSM8K.

> Run with the same conda environment. Notebooks assume the evaluation scripts have produced artifacts in the default output locations.

---

## Expected Outputs

* **Tuned‑Lens**: layer‑wise top‑1 pseudo‑logits, calibration metrics, and per‑layer accuracy deltas (saved as CSV/Parquet and plotted as heatmaps).
* **IG**: token‑level attributions split by **prompt** vs **reasoning** spans; summary statistics across the dataset; per‑example plots.

Paths are configurable in the scripts; by default, logs go to `logs/` if you use the commands above.

---

## Hardware & Runtime

* TL (base/GRPO) typically fits on **80GB** GPUs at the given sequence lengths; use `--quantized` and/or **shorter** `--max_*tokens` if constrained.
* IG is slower (multiple attribution steps per token). Start with `--n-steps 10` for smoke tests.

---

## Acknowledgements

The authors gratefully acknowledge the Instituto Tecnológico de Aragón (ITA) for providing high-performance-computing resources and expert technical support. This research was funded by the Spanish Ministry of Science and Innovation through the 2022 Public–Private Collaboration Programme, grant CPP2022-009790 (MAGELLAN: Design and deployment of a SaaS behavioural engine based on the ETR® AGI architecture).

All product names, trademarks and registered trademarks are property of their respective owners; their use herein does not imply endorsement.

<p align="center">
  <img src="docs/logos/funding_logos.jpg" alt="Ministerio de Ciencia, Innovación y Universidades · Financiado por la Unión Europea NextGenerationEU · Plan de Recuperación, Transformación y Resiliencia · Agencia Estatal de Investigación" width="700">
</p>

---

## License

© 2026 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).

Third‑party components retain their original licenses.

---
