#!/bin/bash

#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

# export CUDA_VISIBLE_DEVICES=0

wandb disabled
wandb login --relogin eef5300e979fc6997734b2e8d2f9d59c6d975d0d

accelerate launch --num_machines=1 --num_processes=2 --dynamo_backend=inductor --use_deepspeed --zero_stage=2 \
 --main_process_port=50001 --mixed_precision=bf16 \
 main.py --project="template" --run_name="Experiemnt1" --model="google/gemma-7b" --batch_size=2 \
 --log_every=500 --bf16 