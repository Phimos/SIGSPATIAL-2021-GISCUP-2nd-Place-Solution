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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import GISCUPDataset, collate_fn
from model import MAPE, RMSPE, WDR


class GISCUPModel(pl.LightningModule):
    def __init__(
        self,
        driver_num,
        link_num,
        wide_config,
        deep_config,
        rnn_config,
        lr: float = 0.0001,
        weight_decay=0.0,
        submission_file: str = "submission.csv",
    ):
        super(GISCUPModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = WDR(
            driver_num=driver_num,
            link_num=link_num,
            wide_config=wide_config,
            deep_config=deep_config,
            rnn_config=rnn_config,
        )
        self.loss = MAPE()
        self.metric = MAPE()

        self.submission_file = submission_file

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, train_batch, batch_idx):
        (
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
            seq_label,
            label,
            weight,
            order_id,
        ) = train_batch
        pred, arrival_pred, _ = self.forward(
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
        )
        packed_label = pack_padded_sequence(
            seq_label, link_len, batch_first=True, enforce_sorted=False
        ).data.squeeze()
        arrival_pred = arrival_pred[packed_label > 0]
        packed_label = packed_label[packed_label > 0] - 1
        anx_loss = F.cross_entropy(arrival_pred, packed_label)
        mape_loss = self.loss(pred, label, weight)
        self.log("train_anx_loss", anx_loss)
        self.log("train_mape_loss", mape_loss)
        loss = mape_loss + 0.1 * anx_loss
        self.log("train_loss", loss)

        if self.global_step % 4096 == 0:
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
            seq_label,
            label,
            weight,
            order_id,
        ) = val_batch
        pred, *_ = self.forward(
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
        )

        loss = self.loss(pred, label)
        mape = self.metric(pred, label)
        self.log("val_loss", loss)
        self.log("val_mape", mape)

    def test_step(self, test_batch, batch_idx):
        (
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
            seq_label,
            label,
            weight,
            order_id,
        ) = test_batch
        pred, *_ = self.forward(
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
        )
        pred = pred.detach().cpu().numpy().reshape(-1, 1) * 1000.0
        order_id = np.array(order_id).reshape(-1, 1)

        data = np.concatenate([order_id, pred], axis=1)
        df = pd.DataFrame(data, columns=["id", "result"])
        return df

    def test_epoch_end(self, outputs):
        submit = pd.concat(outputs, ignore_index=True)
        submit.to_csv(self.submission_file, index=False)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
