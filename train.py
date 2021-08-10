import os
from pprint import pprint
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import GISCUPDataset, Tokenizer, collate_fn
from plmodel import GISCUPModel

data_dir = "/data3/ganyunchong/giscup_2021"
train_dir = os.path.join(data_dir, "train")
test_file = os.path.join(data_dir, "20200901_test.txt")

GPUS = 1
EPOCHS = 25
WORKERS = 32
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
FOLD = 5

train_end: str = "20200824"
validation_end: str = "20200831"

load_ckpt: str = ""
tokenizer_dir = "/nvme/ganyunchong/didi/10fold"
kfold_data_dir = "/nvme/ganyunchong/didi/10fold"


def train():
    pl.seed_everything(42)

    cpu_num = os.cpu_count()
    assert isinstance(cpu_num, int)

    train_dataset = GISCUPDataset(
        dataset_type="train",
        train_end=train_end,
        validation_end=validation_end,
        fold=FOLD,
        tokenizer_dir=tokenizer_dir,
        kfold_data_dir=kfold_data_dir,
    )
    val_dataset = GISCUPDataset(
        dataset_type="val",
        train_end=train_end,
        validation_end=validation_end,
        fold=FOLD,
        tokenizer_dir=tokenizer_dir,
        kfold_data_dir=kfold_data_dir,
    )
    test_dataset = GISCUPDataset(
        dataset_type="test",
        fold=FOLD,
        tokenizer_dir=tokenizer_dir,
        kfold_data_dir=kfold_data_dir,
    )
    train_dataset.load_tokenizer()
    val_dataset.load_tokenizer()
    test_dataset.load_tokenizer()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=collate_fn,
    )

    print("train_step:", len(train_loader))
    print("validation step:", len(val_loader))
    print("test step:", len(test_loader))

    if len(val_loader) == 0:
        val_loader = None

    basic_info, wide_config, deep_config, rnn_config = train_dataset.generate_config()
    pprint(basic_info)
    pprint(wide_config)
    pprint(deep_config)
    pprint(rnn_config)

    if FOLD != 0:
        submission_file = "submission_fold%d.csv" % FOLD
    else:
        submission_file = "submission.csv"

    model = GISCUPModel(
        driver_num=basic_info["driver_num"],
        link_num=basic_info["link_num"],
        wide_config=wide_config,
        deep_config=deep_config,
        rnn_config=rnn_config,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        submission_file=submission_file,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mape",
        mode="min",
        filename="{epoch:02d}-{val_mape:.5f}",
        save_top_k=5,
        verbose=True,
    )

    trainer = pl.Trainer(
        gpus=GPUS,
        max_epochs=EPOCHS,
        benchmark=True,
        deterministic=True,
        stochastic_weight_avg=True,
        distributed_backend="ddp" if GPUS > 1 else None,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=load_ckpt if load_ckpt != "" else None,
    )

    trainer.fit(model, train_loader, val_loader)

    model = GISCUPModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        driver_num=basic_info["driver_num"],
        link_num=basic_info["link_num"],
        wide_config=wide_config,
        deep_config=deep_config,
        rnn_config=rnn_config,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        submission_file=submission_file,
    )

    trainer.test(model, test_loader)


if __name__ == "__main__":
    train()
