#!/bin/bash

 
set -euo pipefail
DRY=${1:-}
SEEDS="42 43 44"
TASKS="regression classification"
 

CONFIGS=(
  "01_frozen_linear   | readout=linear readout.dropout=0 train.weight_decay=0"
  "01_frozen_mlp      | readout=mlp    readout.dropout=0 train.weight_decay=0"
 
  # unfreeze
  "02_unfrozen_linear | readout=linear readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false"
  "02_unfrozen_mlp    | readout=mlp    readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false"
 
  #  warmup
  "03_warmup_lr1e-5_linear | readout=linear readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-5"
  "03_warmup_lr1e-5_mlp    | readout=mlp    readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-5"
  "03_warmup_lr1e-4_linear | readout=linear readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
  "03_warmup_lr1e-4_mlp    | readout=mlp    readout.dropout=0 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
 
  # dropout in Mlp
  "04_dropout01_mlp   | readout=mlp readout.dropout=0.1 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
  "04_dropout03_mlp   | readout=mlp readout.dropout=0.3 train.weight_decay=0 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
 
  # weight decay
  "05_wd1e-4_mlp      | readout=mlp readout.dropout=0.1 train.weight_decay=1e-4 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
  "05_wd1e-2_mlp      | readout=mlp readout.dropout=0.1 train.weight_decay=1e-2 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4"
 
  # scheduler
  "06_reduce_lr_mlp   | readout=mlp readout.dropout=0.1 train.weight_decay=1e-4 model.freeze_encoder=false train.head_only_epochs=5 model.encoder_lr=1e-4 train.reduce_lr=true"
)
 
n=0
for entry in "${CONFIGS[@]}"; do
  name=$(echo "${entry%%|*}" | xargs)
  overrides=$(echo "${entry#*|}" | xargs)
  for task in $TASKS; do
    for seed in $SEEDS; do
      n=$((n+1))
      full="$overrides experiment_name=r2/$name"
      if [[ "$DRY" == "-n" ]]; then
        echo "$n  $task  seed=$seed  $full"
      else
        MODEL=chemeleon TASK="$task" SEED="$seed" OVERRIDES="$full" bsub < finetune_uv.sh
        sleep 0.3
      fi
    done
  done
done
echo "Total: $n tasks"
 