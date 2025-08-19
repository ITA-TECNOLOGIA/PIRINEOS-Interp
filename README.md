# Characterizing Reasoning Models with Layer‑Wise Metrics from Probing and Attributions

**Code & data for our NeurIPS 2025 Mechanistic Interpretability Workshop submission.**
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

## Quickstart (sanity check in <10 minutes)

> Useful for reviewers: runs a short, quantized pass on the base model to verify the stack.

```bash
# 1) Create & activate environment
conda create -n qwen-tunedlens-ig python=3.11 -y
conda activate qwen-tunedlens-ig

# 2) Install core deps (PyTorch must match your CUDA; example below is CUDA 12.1/12.2)
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3) Initialize submodules (Tuned Lens & Captum fork)
git submodule update --init --recursive

# 4) (If using private or gated models) Set your Hugging Face token securely
export HF_TOKEN=***your_hf_token***

# 5) Run a small quantized IG pass on the base model
export CUDA_VISIBLE_DEVICES=0
python src/main/python/eval_gsm8k/ig_evaluation.py \
  --model Qwen/Qwen2.5-14B \
  --max-new-tokens 256 \
  --n-steps 10 \
  --internal-batch-size 1 \
  --quantized
```

If this finishes and saves outputs (see `logs/` or your terminal), your environment is set.

---

## Full Setup

### 1) Python & CUDA

We used **Python 3.11** and **CUDA 12.2**. A safe baseline:

```bash
conda create -n qwen-tunedlens-ig python=3.11 -y
conda activate qwen-tunedlens-ig

# PyTorch for CUDA 12.1/12.2 (adjust if your CUDA differs)
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu121
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
# If you plan to use Open-R1 utilities
git clone https://github.com/huggingface/open-r1.git
cd open-r1
pip install -e ".[dev]"
cd -

# vLLM (CUDA 12.x wheel index line may already be set by torch above)
pip install -q vllm==0.6.6.post1
pip install -q vllm==0.6.6.post1 --extra-index-url https://download.pytorch.org/whl/cu121

# Project requirements
pip install -r requirements.txt

# Tuned Lens from PyPI + Plotly pin for heatmaps
pip install tuned-lens
pip install plotly==5.13.1
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
  * RL‑fine‑tuned (GRPO): `outputs/Qwen2_5_14B_GRPO/checkpoint-1500`
  * Distilled: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

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

## License

Third‑party components retain their original licenses.

---
