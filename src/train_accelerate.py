"""Module to train the Model"""

import json
import logging
import os

import torch
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

    def __init__(self, args, accelerator) -> None:
        self.args = args
        self.accelerator = accelerator

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

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )

    def train(self):
        """Train the model"""
        running_loss = 0

        self.model.train()
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, attention_mask, labels = (
                    batch["input_ids"][:, :-1].contiguous().to(self.model.device),
                    batch["attention_mask"][:, :-1].contiguous().to(self.model.device),
                    batch["labels"][:, 1:].contiguous().to(self.model.device),
                )

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )["logits"]
                B, S, V = outputs.shape

                loss = self.critation(outputs.reshape(B * S, V), labels.reshape(B * S))
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()

                if i % self.args.log_every == self.args.log_every - 1:
                    perplexity = self.validate()
                    running_loss /= self.args.log_every

                    self.accelerator.log(
                        {
                            "lr": self.scheduler.get_lr()[0],
                            "running_loss": running_loss,
                            "perplexity": perplexity,
                        },
                        step=epoch * len(self.train_loader) + i,
                    )

                    running_loss = 0

            logger.info("Status: Saving Model")
            self.save_pretrained(os.path.join(self.path, f"epoch_{epoch}"))

    @torch.no_grad()
    def validate(self):
        """Calculates the perplexity on Validation dataset"""
        nlls = []
        for i, val_batch in enumerate(self.train_loader):
            input_ids, attention_mask, labels = (
                val_batch["input_ids"][:, :-1].contiguous().to(self.model.device),
                val_batch["attention_mask"][:, :-1].contiguous().to(self.model.device),
                val_batch["labels"][:, 1:].contiguous().to(self.model.device),
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
