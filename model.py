import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class MAPE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        ape = (pred - target).abs() / (target.abs() + self.eps)
        if weight is None:
            return torch.mean(ape)
        else:
            return (ape * weight / (weight.sum())).sum()


class RMSPE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.sqrt(
            torch.mean(torch.square((pred - target).abs() / (target.abs() + self.eps)))
        )


class Wide(nn.Module):
    def __init__(self, wide_config):
        super(Wide, self).__init__()
        self.config = wide_config
        self.linear = nn.Linear(in_features=self.feature_dim, out_features=256)

    @property
    def feature_dim(self) -> int:
        dense_dim = self.config["dense"]["size"]
        sparse_dim = sum([feature["size"] for feature in self.config["sparse"]])
        cross_dim = sum(
            [
                feature1["size"] * feature2["size"]
                for feature1, feature2 in itertools.combinations(
                    self.config["sparse"], 2
                )
            ]
        )
        return dense_dim + sparse_dim + cross_dim

    def forward(self, dense, sparse):
        sparse_features = []

        for feature in self.config["sparse"]:
            onehot = F.one_hot(sparse[..., feature["col"]], feature["size"])
            sparse_features.append(onehot)

        for feature1, feature2 in itertools.combinations(self.config["sparse"], 2):
            cross = (
                sparse[..., feature1["col"]] * feature2["size"]
                + sparse[..., feature2["col"]]
            )

            onehot = F.one_hot(cross, feature1["size"] * feature2["size"])
            sparse_features.append(onehot)

        sparse_feature = torch.cat(sparse_features, -1)

        features = torch.cat([dense, sparse_feature], -1)

        out = self.linear(features)
        out = F.relu(out)

        return out


class Deep(nn.Module):
    def __init__(self, deep_config, dropout=0.1):
        super(Deep, self).__init__()
        self.config = deep_config

        for feature in self.config["sparse"]:
            setattr(
                self,
                "embedding_%s" % feature["name"],
                nn.Embedding(feature["size"], feature["dim"]),
            )

        self.linear1 = nn.Linear(self.feature_dim, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, 256)

    @property
    def feature_dim(self) -> int:
        dim = self.config["dense"]["size"]
        for feature in self.config["sparse"]:
            dim += feature["dim"]
        return dim

    def forward(self, dense, sparse):
        sparse_features = []

        for feature in self.config["sparse"]:
            embed = getattr(self, "embedding_%s" % feature["name"])(
                sparse[..., feature["col"]]
            )
            sparse_features.append(embed)

        sparse_feature = torch.cat(sparse_features, -1)

        features = torch.cat([dense, sparse_feature], -1)

        out = self.linear1(features)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = F.relu(out)
        return out, features


class Recurrent(nn.Module):
    def __init__(
        self,
        link_num,
        embedding_dim=20,
        dense_dim=7,
        hidden_dim=256,
        query_dim=109,
        dropout=0.1,
        rnn_config=None,
    ):
        super().__init__()
        self.config = rnn_config

        for feature in self.config["sparse"]:
            setattr(
                self,
                "embedding_%s" % feature["name"],
                nn.Embedding(feature["size"], feature["dim"]),
            )

        self.link_embedding = nn.Embedding(link_num, embedding_dim)
        self.linear_link = nn.Linear(embedding_dim + self.feature_dim, hidden_dim)
        self.linear_cross = nn.Linear(embedding_dim * 2 + 1, hidden_dim)
        self.lstm_link = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_cross = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        self.arrival_pred_head = nn.Linear(hidden_dim, 5)

    @property
    def feature_dim(self) -> int:
        dim = self.config["dense"]["size"]
        for feature in self.config["sparse"]:
            dim += feature["dim"]
        return dim

    def forward(
        self,
        seq_dense,
        seq_sparse,
        link_id,
        link_len,
        cross_id_start=None,
        cross_id_end=None,
        cross_dense=None,
        cross_len=None,
        query=None,
    ):
        sparse_features = []

        for feature in self.config["sparse"]:
            embed = getattr(self, "embedding_%s" % feature["name"])(
                seq_sparse[..., feature["col"]]
            )
            sparse_features.append(embed)

        sparse_feature = torch.cat(sparse_features, -1)

        embed = self.link_embedding(link_id)

        embed_cross_start = self.link_embedding(cross_id_start)
        embed_cross_end = self.link_embedding(cross_id_end)

        seq = torch.cat([seq_dense, embed, sparse_feature], dim=-1)

        seq = self.linear_link(seq)
        seq = F.relu(seq)

        cross = torch.cat([cross_dense, embed_cross_start, embed_cross_end], dim=-1)
        cross = self.linear_cross(cross)
        cross = F.relu(cross)

        packed_link = pack_padded_sequence(
            seq, link_len, batch_first=True, enforce_sorted=False
        )
        out_link, (ht_link, _) = self.lstm_link(packed_link)

        packed_cross = pack_padded_sequence(
            cross, cross_len, batch_first=True, enforce_sorted=False
        )
        _, (ht_cross, _) = self.lstm_cross(packed_cross)

        arrival_pred = self.arrival_pred_head(out_link.data)

        return torch.cat([ht_link[-1], ht_cross[-1]], dim=-1), arrival_pred


class WDR(nn.Module):
    def __init__(
        self,
        driver_num,
        link_num,
        embedding_dim: int = 20,
        wide_config=None,
        deep_config=None,
        rnn_config=None,
        dropout=0.1,
    ):
        super(WDR, self).__init__()
        self.wide = Wide(wide_config)
        self.deep = Deep(deep_config, dropout=dropout)
        self.recurrent = Recurrent(
            link_num=link_num, dropout=dropout, rnn_config=rnn_config
        )

        self.head = nn.Linear(256 + 256 + 256 + 256, 32)
        self.head2 = nn.Linear(32, 1)

    def forward(
        self,
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
    ):

        out_wide = self.wide(dense, sparse)
        out_deep, feature_deep = self.deep(dense, sparse)
        out_recurrent, arrival_pred = self.recurrent(
            seq_dense,
            seq_sparse,
            link_id,
            link_len,
            cross_id_start,
            cross_id_end,
            cross_dense,
            cross_len,
            feature_deep,
        )
        features = torch.cat([out_wide, out_deep, out_recurrent], -1)
        out = self.head(features)
        out = F.relu(out)
        out = self.head2(out)
        return out, arrival_pred, features
