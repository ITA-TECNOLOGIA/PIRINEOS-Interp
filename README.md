# Q-AGIITA borraja

## Qwen conda env

```bash
conda create -n qwen-ig python=3.11 -y
conda activate qwen-ig
```

Now required libraries for running qwen (and maybe further training)
```bash
git clone https://github.com/huggingface/open-r1.git
cd open-r1/
pip install -e ".[dev]"
pip install vllm==0.6.6.post1 -q
pip install vllm==0.6.6.post1 --extra-index-url https://download.pytorch.org/whl/cu121 -q
```

Install pytorch for current CUDA (12.2) -> torch 2.6.0, otherwise you won't be able to import torch. You'll find a warning from vllm and xformers installations.
```bash
pip install torch==2.6.0 torchvision==0.21.0
```


## Captum submodule

```bash
git clone https://github.com/plazaro-ita/captum-ITA.git
git submodule init
git submodule update
git submodule add https://github.com/plazaro-ita/captum-ITA.git captum-ITA/
```

Install requirements for captum
```bash
cd captum-ITA
pip install -e .
```


