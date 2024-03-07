"""Moduile to Load dataset and create Dataloader for Training"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator


def prepare_dataloader(args, tokenizer):
    """Load dataset and Prepare Dataloader"""

    def pre_process(item):
        processed = tokenizer(
            item["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
            pad_to_multiple_of=2,
        )

        labels = processed["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        processed["labels"] = labels
        return processed

    dataset = load_dataset(args.dataset)

    dataset["train"] = dataset["train"].select(range(5000))

    dataset = dataset.map(
        pre_process,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader
