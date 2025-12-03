![Python](https://img.shields.io/badge/python-3.10%2B-blue) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org/)


# Bridging Discourse Treebanks with a Unified RST Parser

Official code for the [CODI 2025 paper](https://aclanthology.org/2025.codi-1.17/) ([Slides](slides_codi25.pdf)).

## Inference

Refer to [IsaNLP RST](https://github.com/tchewik/isanlp_rst) for running the trained multilingual parser.

## Experiments
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu121 torch
````


### Monolingual (any corpus, or all)

List available corpora:

```bash
bash src/commands.sh list
```

```bash
# single corpus
bash src/commands.sh mono train eng.erst.gum
# run across all corpora used in the unified setups
bash src/commands.sh mono train all
# filter when running "all"
ONLY='eng|rus' bash src/commands.sh mono evaluate all
# data augmentation toggle
AUG=true bash src/commands.sh mono train eng.rst.rstdt
```

### Unified setups

```bash
# UniRST MuH. mseg = multiple segmentation heads, sseg = single head.
bash src/commands.sh unir-muh-mseg train
bash src/commands.sh unir-muh-mseg evaluate

# UniRST UU
bash src/commands.sh unir-uu train
bash src/commands.sh unir-uu evaluate

# UniRST MU (default).
bash src/commands.sh unir-mu-mseg train
bash src/commands.sh unir-mu-mseg evaluate
```

### Common toggles (env vars)

```bash
RUNS=3        # number of random restarts
CUDA=0        # select GPU (maps to --cuda_device)
AUG=true      # enable data augmentation
SSEG=true     # single segmentation head (MuH/MU); false = multiple heads
DEBUG=1       # print shell commands
```
