#!/bin/bash

#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

# export CUDA_VISIBLE_DEVICES=0

wandb disabled
wandb login --relogin eef5300e979fc6997734b2e8d2f9d59c6d975d0d

python main.py --project="template" --run_name="Experiemnt1" --model="gpt2" --batch_size=2 \
 --log_every=500 --bf16