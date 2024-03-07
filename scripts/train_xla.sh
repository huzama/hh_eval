#!/bin/bash

wandb disabled
wandb login --relogin eef5300e979fc6997734b2e8d2f9d59c6d975d0d

export PJRT_ALLOCATOR_FRACTION=0.98

python main.py --project="template" --run_name="Experiemnt1" --model="google/gemma-7b" --batch_size=16 \
 --log_every=500 --bf16 --xla 