"""Module containing helper functions for other modules"""

import argparse
import json
import logging
import os
import time

import torch
from accelerate import Accelerator
from openai import OpenAI
from safetensors.torch import load_model as load_model_safetensors

logger = logging.getLogger(__name__)
########################################################################################


def load_model(model_class, path):
    with open(os.path.join(path, "args.json"), "r") as f:
        _args = json.load(f)
        parser = argparse.ArgumentParser()
        args = parser.parse_args(args=[])
        args.__dict__.update(_args)

    model = model_class(args)
    if args.compile:
        model = torch.compile(model)

    load_model_safetensors(model, os.path.join(path, "model.safetensors"), strict=True)

    return args, model, model.tokenizer


########################################################################################


def print_stats(model, batch_size, seq_len):
    # https://erees.dev/transformer-memory/
    # Get number of trainable parameters from model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / (
        1024**2
    )

    logger.info(f"Number of Trainable Parameters: {num_params:.2f}M")

    # Compute and log the size in MB of the model in memory
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024**3
    )

    optimizer_size_gb = 2 * (
        sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        / (1024**3)
    )
    gradient_size_gb = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    ) / (1024**3)

    input_size_gb = batch_size * seq_len * 8 / (1024**3)

    n_e = batch_size * seq_len * model.config.hidden_size
    n_a = batch_size * seq_len * seq_len * model.config.num_attention_heads
    n_l = batch_size * seq_len * model.config.vocab_size

    activation_size_gb = (
        0.9 * model.config.num_hidden_layers * (36 * n_e + 6 * n_a) + 6 * n_e + 6 * n_l
    ) / (1024**3)

    logger.info(
        f"Model Size: {model_size_gb:.2f}GB, Optimizer Size: {optimizer_size_gb:.2f}GB, Gradient Size: {gradient_size_gb:.2f}GB, Activation Size: {activation_size_gb:.2f}GB, Total Size: {input_size_gb + model_size_gb + optimizer_size_gb + gradient_size_gb + activation_size_gb:.2f}GB"
    )


########################################################################################

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def message_format(message, color):
    if color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

FORMAT = "%(asctime)s [$BOLD %(name)s $RESET][ %(levelname)s ] %(message)s (%(filename)s:%(lineno)d)"
COLOR_FORMAT = message_format(FORMAT, True)


COLORS = {
    "WARNING": YELLOW,
    "INFO": BLUE,
    "DEBUG": WHITE,
    "CRITICAL": RED,
    "ERROR": RED,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            )
            record.levelname = levelname_color
            record.msg = COLOR_SEQ % (30 + COLORS[levelname]) + record.msg + RESET_SEQ
        return logging.Formatter.format(self, record)


def setup_logging(log_level=os.environ.get("LOGLEVEL", "INFO")):
    color_formatter = ColoredFormatter(COLOR_FORMAT)
    h = logging.StreamHandler(None)
    h.setFormatter(color_formatter)
    logging.root.addHandler(h)
    logging.root.setLevel(log_level)


def initialize_accelerator(args):
    """Function to initialize distributed training"""

    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerator.init_trackers(
        args.project,
        init_kwargs={
            "wandb": {
                "entity": args.entity,
                "name": f"{args.run_name}_{args.model.split('/')[-1]}{'(rand)' if args.random_init else ''}_{args.dataset}_{args.start_id}",
                "config": args.__dict__,
                "resume": args.resume,
                "id": args.start_id,
            }
        },
    )

    return accelerator


def generate_openai(args, prompt, system):
    """Generate function for OpenAI GPT-4 Model"""
    client = OpenAI(api_key=args.openai_key)
    for _ in range(200):
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system
                            if system is not None
                            else "You are a helpful assistant."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=args.num_return_sequences,
            )
            break
        except RuntimeError:
            time.sleep(2)

    outputs = [c.message.content for c in response.choices]
    if len(outputs) == 1:
        return outputs[0]
    return outputs


########################################################################################


def get_postion_ids(input_ids, tokenizer):
    position_ids = torch.zeros_like(input_ids)
    for i, ids in enumerate(input_ids):
        pad_len = torch.nonzero(ids != tokenizer.pad_token_id, as_tuple=True)[0][0]
        non_pad_len = len(input_ids[0]) - pad_len
        position_ids[i, pad_len:] = torch.arange(non_pad_len)

    return position_ids


########################################################################################


@torch.no_grad()
def generate(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    """Generate function for Huggingface models"""

    max_new_tokens = kwargs.pop("max_new_tokens", 128)
    tpu = kwargs.pop("TPU", False)

    device = next(model.parameters()).device

    str_input = isinstance(input_ids, str)

    if str_input:
        inputs = tokenizer([input_ids], return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

    B, S = input_ids.shape

    generation = torch.zeros((B, S + max_new_tokens), dtype=torch.long, device=device)
    generation[:, :S] = input_ids

    if past_key_values is None:
        n_token = 0
    else:
        n_token = past_key_values[0][0].shape[-2]

    _attention_mask = torch.zeros(
        (B, S + n_token + max_new_tokens),
        dtype=torch.long,
        device=device,
    )
    if attention_mask is not None:
        _attention_mask[:, : S + n_token] = attention_mask
    else:
        _attention_mask[:, : S + n_token] = 1

    _position_ids = torch.zeros(
        (B, S + max_new_tokens), dtype=torch.long, device=device
    )
    if position_ids is None:
        _position_ids[:, :S] = torch.arange(S, device=device)
    else:
        _position_ids[:, :S] = position_ids

    if tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.experimental.xla_sharding as xs

        if "spda_mesh" in kwargs:
            xs.mark_sharding(generation, kwargs["spda_mesh"], ("data", None))
            xs.mark_sharding(_attention_mask, kwargs["spda_mesh"], ("data", None))
            xs.mark_sharding(_position_ids, kwargs["spda_mesh"], ("data", None))

    for idx in range(max_new_tokens):
        outputs = model(
            input_ids=generation,
            attention_mask=_attention_mask,
            position_ids=_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs["logits"]

        current_index = torch.tensor([idx + S - 1], device=generation.device)
        next_token_logits = torch.index_select(logits, 1, current_index).squeeze(1)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_index = torch.tensor([idx + S], device=generation.device)

        generation = generation.index_copy(1, next_index, next_token_id)
        _attention_mask = _attention_mask.index_copy(
            1, next_index, torch.ones_like(next_token_id)
        ).to(device)

        position = torch.max(_position_ids, dim=1, keepdim=True).values
        _position_ids = _position_ids.index_copy(1, next_index, position + 1)

        if tpu:
            xm.mark_step()

    generation = generation[:, S:]

    for out in generation:
        eos_idx = torch.where(out == tokenizer.eos_token_id)[0]
        if len(eos_idx) > 0:
            out[eos_idx[0] + 1 :] = tokenizer.eos_token_id

    if str_input:
        return tokenizer.batch_decode(generation, skip_special_tokens=True)
    else:
        return generation


# def generate_cache(self, input_ids, attention_mask=None, **kwargs):
#     """Generate function for Huggingface models"""

#     prefix = kwargs.pop("prefix", None)
#     system = kwargs.pop("system", None)
#     max_new_tokens = kwargs.pop("max_new_tokens", 128)

#     str_input = isinstance(input_ids, str)

#     if str_input:
#         if self.args.model.lower() == "gpt4":
#             return self.generate_openai(input_ids, system)
#         else:
#             inputs = self.tokenizer([input_ids], return_tensors="pt")
#             input_ids = inputs["input_ids"].to(self.device)
#             attention_mask = inputs["attention_mask"].to(self.device)

#     _, S = input_ids.shape

#     # Precompute Past Key Values
#     if prefix is not None:
#         outs = self.model_gen(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             prefix=prefix,
#             use_cache=True,
#             past_key_values=None,
#         )
#     else:
#         outs = self.model_gen(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=True,
#             past_key_values=None,
#         )

#     past_key_values = outs.past_key_values
#     next_token = torch.argmax(outs.logits[:, -1, :], dim=-1, keepdim=True)
#     attention_mask = torch.ones_like(next_token)
#     input_ids = torch.cat((input_ids, next_token), dim=-1)

#     # Generating sequence
#     for _ in range(max_new_tokens - 1):
#         if prefix is not None:
#             outs = self.model_gen(
#                 input_ids=next_token,
#                 attention_mask=attention_mask,
#                 prefix=prefix,
#                 use_cache=True,
#                 past_key_values=past_key_values,
#             )
#         else:
#             outs = self.model_gen(
#                 input_ids=next_token,
#                 attention_mask=attention_mask,
#                 use_cache=True,
#                 past_key_values=past_key_values,
#             )

#         past_key_values = outs.past_key_values

#         next_token = torch.argmax(outs.logits[:, -1, :], dim=-1, keepdim=True)
#         attention_mask = torch.ones_like(next_token)

#         input_ids = torch.cat((input_ids, next_token), dim=-1)

#     generation = input_ids[:, S:]

#     for out in generation:
#         eos_idx = torch.where(out == self.tokenizer.eos_token_id)[0]
#         if len(eos_idx) > 0:
#             out[eos_idx[0] + 1 :] = self.tokenizer.eos_token_id

#     if str_input:
#         return self.tokenizer.batch_decode(generation, skip_special_tokens=True)
#     else:
#         return generation


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.experimental.xla_sharding as xs
    import torch_xla.runtime as xr
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    xr.use_spmd()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16
    )

    model.to(xm.xla_device())

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh_shape = (2, 4)

    spda_mesh = xs.Mesh(device_ids, mesh_shape, ("model", "data"))

    for name, param in model.named_parameters():
        if "embed_tokens" in name:
            xs.mark_sharding(param, spda_mesh, ("model", "data"))
        elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
            xs.mark_sharding(param, spda_mesh, ("data", "model"))
        elif "o_proj" in name:
            xs.mark_sharding(param, spda_mesh, ("model", "data"))
        elif "gate_proj" in name or "up_proj" in name:
            xs.mark_sharding(param, spda_mesh, ("model", "data"))
        elif "down_proj" in name:
            xs.mark_sharding(param, spda_mesh, ("data", "model"))
        elif "lm_head" in name:
            xs.mark_sharding(param, spda_mesh, ("model", "data"))

    input_ids = torch.randint(
        8, 1024, (8, 1536), dtype=torch.long, device=xm.xla_device()
    )

    atten_mask = torch.randint(
        0, 1, (8, 1536), dtype=torch.long, device=xm.xla_device()
    )

    xs.mark_sharding(input_ids, spda_mesh, ("data", None))

    for i in tqdm(range(100)):
        generate(
            model,
            tokenizer,
            input_ids,
            attention_mask=atten_mask,
            TPU=True,
            spda_mesh=spda_mesh,
        )
