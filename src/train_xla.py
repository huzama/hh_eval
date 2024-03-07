"""Module to train the Model"""

import json
import logging
import os

import numpy as np
import torch
import torch.distributed.checkpoint as dist_cp
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.experimental.distributed_checkpoint as xc
import torch_xla.runtime as xr
import wandb
from evaluate import load
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.dataset import prepare_dataloader
from src.utils import print_stats

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer Class"""

    def __init__(self, args) -> None:
        self.args = args

        if self.args.bf16:
            self.dtype = torch.bfloat16
        elif self.args.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info("Status: Loading Model")
        model_args = {
            "pretrained_model_name_or_path": self.args.model,
            "cache_dir": (
                self.args.cache_dir
                if self.args.cache_dir and os.path.exists(self.args.cache_dir)
                else None
            ),
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(**model_args)

        logger.info(f"Status: Initalizing XLA Devices")
        self.device = xm.xla_device()
        self.model.to(self.device)

        self.num_devices = xr.global_runtime_device_count()
        logger.info(f"Number of Devices: {self.num_devices}")

        self.device_ids = np.array(range(self.num_devices))
        mesh_shape = (2, 2)

        self.spda_mesh = xs.Mesh(self.device_ids, mesh_shape, ("model", "data"))

        self.data_shard = xs.ShardingSpec(self.spda_mesh, ("data", None))

        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                xs.mark_sharding(param, self.spda_mesh, ("model", "data"))
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                xs.mark_sharding(param, self.spda_mesh, ("data", "model"))
            elif "o_proj" in name:
                xs.mark_sharding(param, self.spda_mesh, ("model", "data"))
            elif "gate_proj" in name or "up_proj" in name:
                xs.mark_sharding(param, self.spda_mesh, ("model", "data"))
            elif "down_proj" in name:
                xs.mark_sharding(param, self.spda_mesh, ("data", "model"))
            elif "lm_head" in name:
                xs.mark_sharding(param, self.spda_mesh, ("model", "data"))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.critation = torch.nn.CrossEntropyLoss()

        print_stats(
            self.model,
            args.batch_size,
            args.max_length,
        )

        tokenizer_args = {
            "pretrained_model_name_or_path": self.args.model,
            "cache_dir": (
                self.args.cache_dir
                if self.args.cache_dir and os.path.exists(self.args.cache_dir)
                else None
            ),
        }

        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.unk_token_id
        self.tokenizer.padding_side = "left"

        logger.info("Status: Loading Dataloader")
        self.train_loader = prepare_dataloader(args, self.tokenizer)

        total_steps = self.args.epochs * len(self.train_loader)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            int(total_steps * 0.1),
            total_steps,
        )

        self.bertscore = load("bertscore")
        self.rouge = load("rouge")

        self.path = os.path.join(
            self.args.save_dir,
            f"{self.args.run_name}_{self.args.model.split('/')[-1]}_{self.args.dataset}_{self.args.start_id}",
        )

        os.makedirs(self.path, exist_ok=True)

    def train(self):
        """Train the model"""
        logger.info("Status: Training Model")

        running_loss = 0
        self.model.train()
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(
                tqdm(
                    pl.MpDeviceLoader(
                        self.train_loader, self.device, input_sharding=self.data_shard
                    )
                )
            ):
                input_ids, attention_mask, labels = (
                    batch["input_ids"][:, :-1],
                    batch["attention_mask"][:, :-1],
                    batch["labels"][:, 1:],
                )

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )["logits"]
                B, S, V = outputs.shape

                loss = self.critation(outputs.reshape(B * S, V), labels.reshape(B * S))

                loss.backward()
                xm.optimizer_step(self.optimizer)
                self.scheduler.step()

                running_loss += loss.item()

                if i % self.args.log_every == self.args.log_every - 1:
                    perplexity = self.validate()
                    running_loss /= self.args.log_every

                    if xm.is_master_ordinal(local=False) and xr.host_index() == 0:
                        wandb.log(
                            {
                                "lr": self.scheduler.get_lr()[0],
                                "running_loss": running_loss,
                                "perplexity": perplexity,
                            },
                            step=epoch * len(self.train_loader) + i,
                        )

                        xp.trace_detached(
                            "localhost:9012", os.path.join(self.path, "logs")
                        )

                    running_loss = 0

            logger.info("Status: Saving Model")
            self.save_pretrained(os.path.join(self.path, f"epoch_{epoch}"))

    @torch.no_grad()
    def validate(self):
        """Calculates the perplexity on Validation dataset"""
        nlls = []
        for i, val_batch in enumerate(
            pl.MpDeviceLoader(
                self.train_loader, self.device, input_sharding=self.data_shard
            )
        ):
            input_ids, attention_mask, labels = (
                val_batch["input_ids"][:, :-1],
                val_batch["attention_mask"][:, :-1],
                val_batch["labels"][:, 1:],
            )

            # Evaluation here
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                "logits"
            ]
            B, S, V = outputs.shape

            loss = self.critation(outputs.reshape(B * S, V), labels.reshape(B * S))

            nlls.append(loss)

            if i == 100:
                break

        return torch.exp(torch.stack(nlls).mean())

    def save_pretrained(self, path):
        "Save the pretrained model to the path"

        self.accelerator.save_model(self.model, path, max_shard_size=int(1e12))

        if self.accelerator.is_local_main_process:
            with open(os.path.join(path, "args.json"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)
