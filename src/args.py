"""Module to define launch arguments"""

import argparse
import sys
import time


def get_args(_args=None):
    """Module to add launch argument to the script"""
    parser = argparse.ArgumentParser()

    # Run Details
    started_at = time.gmtime()
    start_id = f"{started_at.tm_year%100}{started_at.tm_mon:02d}{started_at.tm_mday:02d}{started_at.tm_hour:02d}{started_at.tm_min:02d}"
    parser.add_argument("--start_id", default=start_id)

    # Model parameters
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--openai_key", default=None, type=int)

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_return_sequences", type=int, default=None)
    parser.add_argument("--temperature", type=int, default=None)

    # Dataset
    parser.add_argument("--dataset", default="bookcorpus", type=str)
    parser.add_argument("--max_length", default=512, type=int)

    # Training parameters
    parser.add_argument("--xla", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", default=None, type=str)

    # Wandb Parameters
    parser.add_argument("--run_name", default="Experiment1", type=str)
    parser.add_argument("--project", default="template", type=str)
    parser.add_argument("--entity", default="huzama")

    # Save Dir
    parser.add_argument("--save_dir", type=str, default="saved")

    ## Setting parameters for fast debugging
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(args=_args)

    # If debugger is attached, set debug to true
    setattr(args, "debug", hasattr(sys, "gettrace") and sys.gettrace() is not None)

    return args
