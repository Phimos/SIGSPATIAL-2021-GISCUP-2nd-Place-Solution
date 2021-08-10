import os
import random
import time
from pprint import pprint
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import GISCUPDataset, Tokenizer, collate_fn
from model import MAPE, WDR
from plmodel import GISCUPModel

data_dir = "/data3/ganyunchong/giscup_2021"
train_dir = os.path.join(data_dir, "train")
test_file = os.path.join(data_dir, "20200901_test.txt")

GPUS = 1
EPOCHS = 30
WORKERS = 32
BATCH_SIZE = 1024
FOLD = 1

train_end: str = "20200824"
validation_end: str = "20200831"

ckpt_path = "/nvme/ganyunchong/didi/lightning_logs/version_194/checkpoints/epoch=19-val_mape=0.11666.ckpt"
tokenizer_dir = "/nvme/ganyunchong/didi/kfold"


def test():
    pl.seed_everything(42)

    cpu_num = os.cpu_count()
    assert isinstance(cpu_num, int)

    train_dataset = GISCUPDataset(
        dataset_type="train",
        train_end=train_end,
        validation_end=validation_end,
        fold=FOLD,
        tokenizer_dir=tokenizer_dir,
    )
    val_dataset = GISCUPDataset(
        dataset_type="val",
        train_end=train_end,
        validation_end=validation_end,
        fold=FOLD,
        tokenizer_dir=tokenizer_dir,
    )
    test_dataset = GISCUPDataset(
        dataset_type="test", fold=FOLD, tokenizer_dir=tokenizer_dir
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

    basic_info, wide_config, deep_config, rnn_config = train_dataset.generate_config()
    pprint(basic_info)
    pprint(wide_config)
    pprint(deep_config)
    pprint(rnn_config)

    print("train_step:", len(train_loader))
    print("validation step:", len(val_loader))
    print("test step:", len(test_loader))

    if FOLD != 0:
        submission_file = "submission_fold%d.csv" % FOLD
    else:
        submission_file = "submission.csv"

    model = GISCUPModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        driver_num=basic_info["driver_num"],
        link_num=basic_info["link_num"],
        wide_config=wide_config,
        deep_config=deep_config,
        rnn_config=rnn_config,
        submission_file=submission_file,
    )

    trainer = pl.Trainer(gpus=1, benchmark=True, deterministic=True)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    test()
