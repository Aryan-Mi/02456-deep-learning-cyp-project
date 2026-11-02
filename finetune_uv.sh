#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_finetune_uv
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/cyp_uv_%J.out
#BSUB -e outputs/cyp_uv_%J.err
set -euo pipefail

MODEL="${MODEL:-atom_jepa}"
TASK="${TASK:-regression}"
OVERRIDES="${OVERRIDES:-}"

cd "$LS_SUBCWD"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'
uv run --no-sync python -u finetune.py model="$MODEL" task="$TASK" $OVERRIDES seed=42 device=cuda
