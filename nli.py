import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from datasets import load_dataset

dataset = load_dataset("multi_nli")


# select telephone as source distribution
telephone = dataset.filter(lambda example: example["genre"] == "telephone")
# select fiction subset as target distribution
fiction = dataset.filter(lambda example: example["genre"] == "fiction")
# use the full training set of genres selected as the source domain
train = dataset["train"]
# use the first 10% of the target-domain training set for training
fiction_train = fiction["train"].select(range(int(len(fiction["train"]) * 0.1)))
# use the whole validation set of the source domain
telephone_validation = telephone[
    "validation_matched"
]  # telephone["validation_mismatched"]
# use the first 50% validation set of the target domain as the validation set
fiction_validation = fiction["validation_matched"].select(
    range(int(len(fiction["validation_matched"]) * 0.5))
)

# use the last 50% validation set of the target domain as the test set
fiction_test = fiction["validation_matched"].select(
    range(
        int(len(fiction["validation_matched"]) * 0.5),
        len(fiction["validation_matched"]),
    )
)

print(train.features)

labels = dataset["train"].features["label"].names

print(labels)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def pre_process(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
    )


train = telephone['train'].map(pre_process, batched=True)
test = fiction_test.map(pre_process, batched=True)

train.set_format(type="torch")

test.set_format(type="torch")


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc.item()}

model.to("cuda")

print(model.device)

batch_size = 32

trainer = transformers.Trainer(
    model=model,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    args=transformers.TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2
    ),
)

trainer.train()
trainer.evaluate()