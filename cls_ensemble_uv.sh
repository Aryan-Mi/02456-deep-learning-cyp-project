#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cls_ensemble
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o outputs/cls_ensemble_%J.out
#BSUB -e outputs/cls_ensemble_%J.err
set -euo pipefail

cd "$LS_SUBCWD"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'
uv run --no-sync python -u -m ensembles.cls_ensemble
