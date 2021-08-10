import datetime
import json
import os
import pickle
from multiprocessing import Pool
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from torch import tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

data_dir = "/data3/ganyunchong/giscup_2021"

train_dir = os.path.join(data_dir, "train")
parsed_dir = os.path.join(data_dir, "parsed")
test_file = os.path.join(data_dir, "20200901_test.txt")
final_files = [os.path.join(data_dir, "20200901_test.txt")]
weather_file = os.path.join(data_dir, "weather.csv")
json_dir = os.path.join(data_dir, "json")
tokenizer_dir = os.path.join("/nvme/ganyunchong", "didi", "tokenizer")


def parse_head(head: str):
    order_id, ata, distance, simple_eta, driver_id, slice_id = head.split(" ")
    return {
        "order_id": order_id,
        "ata": float(ata),
        "distance": float(distance),
        "simple_eta": float(simple_eta),
        "driver_id": int(driver_id),
        "slice_id": int(slice_id),
    }


def parse_link(link: str):
    link_id, other_info = link.split(":")
    link_time, link_ratio, link_current_status, link_arrival_status = other_info.split(
        ","
    )
    return {
        "link_id": int(link_id),
        "link_time": float(link_time),
        "link_ratio": float(link_ratio),
        "link_current_status": int(link_current_status),
        "link_arrival_status": int(link_arrival_status),
    }


def parse_links(links: str):
    if links == "":
        return []
    else:
        return [parse_link(link) for link in links.split(" ")]


def parse_cross(cross: str):
    cross_id, cross_time = cross.split(":")
    return {
        "cross_id": cross_id,
        "cross_time": float(cross_time),
    }


def parse_crosses(crosses: str):
    if crosses == "":
        return []
    else:
        return [parse_cross(cross) for cross in crosses.split(" ")]


def parse_order(order: str) -> Dict[str, Any]:
    order = order.replace("\n", "")
    head, link, cross = order.split(";;")
    return {
        "head": parse_head(head),
        "link": parse_links(link),
        "cross": parse_crosses(cross),
    }


class Tokenizer(object):
    def __init__(
        self,
        cat2index: Dict[Any, Any],
        index2cat: Optional[Dict[Any, Any]] = None,
        padding: bool = False,
        default_value: Any = 0,
    ) -> None:
        super().__init__()
        self.cat2index = cat2index
        self.index2cat = index2cat
        self.padding = padding
        self.default_value = default_value

    def encode(self, data: Any) -> Any:
        return self.cat2index.get(data, 0)

    def decode(self, index: Any) -> Any:
        assert self.index2cat is not None
        return self.index2cat[index]

    def __len__(self):
        return len(self.cat2index)


def categorical_to_index(categorical: List[Any], contain_pad=False):
    set_categorical = set(categorical)
    padding = 1 if contain_pad else 0
    categorical2index = {cat: i + padding for i, cat in enumerate(set_categorical)}
    if contain_pad:
        categorical2index["<PAD>"] = 0  # Padding or Unseen
    index2categorical = {v: k for k, v in categorical2index.items()}
    return Tokenizer(categorical2index, index2categorical, contain_pad)


def load_pickle(file_path) -> Any:
    with open(file_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def dump_pickle(data, file_path) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def load_json(file_path) -> Any:
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def dump_json(data, file_path) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, sort_keys=True, indent=4)


def time_weight(order):
    date = order["head"]["date"]
    delta = datetime.datetime(2020, 9, 1) - datetime.datetime(
        int(date[:4]), int(date[4:6]), int(date[6:])
    )
    return 0.98 ** (delta.days - 15)


def time_short_weight(order):
    date = order["head"]["date"]
    delta = datetime.datetime(2020, 9, 1) - datetime.datetime(
        int(date[:4]), int(date[4:6]), int(date[6:])
    )
    short = 1 / np.log(np.clip(order["head"]["distance"] / 1000, 2, 30))
    return 0.98 ** (delta.days - 15) * short


class GISCUPDataset(Dataset):
    def __init__(
        self,
        dataset_type: str = "train",
        train_end: str = "20200824",
        validation_end: str = "20200831",
        transform_func: Optional[Callable] = None,
        tokenizer_dir: str = "/nvme/ganyunchong/didi/tokenizer",
        kfold_data_dir: str = "/nvme/ganyunchong/didi/kfold",
        load: bool = False,
        flush: bool = False,
        fold: int = 0,
        calc_weight=time_weight,
    ):
        super().__init__()

        self.transform_func = transform_func

        self.train_data: List[Any] = []
        self.train_files: List[str] = []
        self.loaded_data: Dict[str, Any] = {}

        self.tokenizer_dir = tokenizer_dir
        self.link_tokenizer = None
        self.driver_tokenizer = None

        self.random_drop = False
        self.load = load
        self.flush = flush
        self.fold = fold

        self.calc_weight = calc_weight

        self.defined_methods = [
            "generate_tokenizer",
            "save_tokenizer",
            "load_tokenizer",
        ]

        assert not (self.load and self.fold > 0)

        if self.load:
            self.weather = pd.read_csv(weather_file)
            self.weather.date = self.weather.date.astype(str)
            self.weather = self.weather.set_index("date")

            filenames = []

            if "train" in dataset_type:
                self.random_drop = True
                for filename in os.listdir(train_dir):
                    date, _ = os.path.splitext(filename)
                    if date <= train_end:
                        filenames.append(os.path.join(train_dir, filename))
            if "val" in dataset_type:
                for filename in os.listdir(train_dir):
                    date, _ = os.path.splitext(filename)
                    if train_end < date <= validation_end:
                        filenames.append(os.path.join(train_dir, filename))
            if "test" in dataset_type:
                filenames += [test_file]

            if "final" in dataset_type:
                filenames += final_files

            filenames.sort()

            for filename in tqdm(filenames):
                date, _ = os.path.splitext(os.path.basename(filename))
                date = date[:8]
                self.loaded_data[date] = self.load_single_file(filename)
                self.train_data += self.loaded_data[date]

        else:
            if fold > 0:
                if "train" in dataset_type:
                    print("load train")
                    self.random_drop = True
                    self.train_files += load_pickle(
                        os.path.join(kfold_data_dir, "fold%d/train_files.pickle" % fold)
                    )
                if "val" in dataset_type:
                    self.train_files += load_pickle(
                        os.path.join(kfold_data_dir, "fold%d/val_files.pickle" % fold)
                    )
                if "test" in dataset_type:
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200901.pickle")
                    )
                if "finetune" in dataset_type:
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200804.pickle")
                    )
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200811.pickle")
                    )
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200818.pickle")
                    )
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200825.pickle")
                    )
            else:
                if "train" in dataset_type:
                    self.random_drop = True
                    for filename in os.listdir(json_dir):
                        date, _ = os.path.splitext(filename)
                        if (
                            os.path.isfile(os.path.join(json_dir, filename))
                            and date <= train_end
                        ):
                            self.train_files += load_pickle(
                                os.path.join(json_dir, filename)
                            )
                if "val" in dataset_type:
                    for filename in os.listdir(json_dir):
                        date, _ = os.path.splitext(filename)
                        if (
                            os.path.isfile(os.path.join(json_dir, filename))
                            and train_end < date <= validation_end
                        ):
                            self.train_files += load_pickle(
                                os.path.join(json_dir, filename)
                            )
                if "test" in dataset_type:
                    self.train_files += load_pickle(
                        os.path.join(json_dir, "20200901.pickle")
                    )

    def load_single_file(self, filepath: str):
        date, _ = os.path.splitext(os.path.basename(filepath))
        date = date[:8]
        pickle_path = os.path.join(parsed_dir, date + ".pickle")

        if os.path.exists(pickle_path) and not self.flush:
            result = load_pickle(pickle_path)
            return result
        else:
            result = []
            year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])
            with open(filepath) as txt_file:
                for order in tqdm(txt_file.readlines()):
                    order_info = parse_order(order)
                    order_info["head"]["weekday"] = datetime.date(
                        year, month, day
                    ).weekday()
                    order_info["head"]["date"] = date
                    order_info["head"]["weather"] = self.weather.loc[date, "weather"]
                    order_info["head"]["hightemp"] = int(
                        self.weather.loc[date, "hightemp"]
                    )
                    order_info["head"]["lowtemp"] = int(
                        self.weather.loc[date, "lowtemp"]
                    )
                    order_info["head"]["json_path"] = self.get_json_path(
                        date, order_info["head"]["order_id"]
                    )
                    result.append(order_info)
            dump_pickle(result, pickle_path)
            return result

    @property
    def df_data(self):
        assert self.load
        df = pd.DataFrame(pd.json_normalize(dataset.train_data))
        return df

    def split_k_fold(self, splits=10, shuffle=True):
        assert self.load
        train_data = self.train_data.copy()
        kfold = KFold(
            n_splits=splits, shuffle=shuffle, random_state=42 if shuffle else None
        )

        for i, (train_idx, val_idx) in enumerate(kfold.split(self.train_data)):
            print("train_idx:", train_idx[:10])
            print("val_idx:", val_idx[:10])
            train_files = [train_data[idx]["head"]["json_path"] for idx in train_idx]
            val_files = [train_data[idx]["head"]["json_path"] for idx in val_idx]
            fold_dir = os.path.join(
                "/nvme/ganyunchong/didi", "10fold", "fold%d" % (i + 1)
            )
            dump_pickle(train_files, os.path.join(fold_dir, "train_files.pickle"))
            dump_pickle(val_files, os.path.join(fold_dir, "val_files.pickle"))
            self.train_data = [train_data[idx] for idx in train_idx]
            self.generate_tokenizer(fold_dir)
            print("FOLD%d Generated!" % (i + 1))
        self.train_data = train_data.copy()

    def __len__(self):
        if self.load:
            return len(self.train_data)
        else:
            return len(self.train_files)

    def __getitem__(self, index):
        if self.load:
            data = self.train_data[index]
        else:
            data = load_json(self.train_files[index])

        if self.transform_func:
            return self.transform_func(data)
        else:
            return self.extract_feature(data)

    @staticmethod
    def get_json_path(date, order_id: str):
        order_id = order_id.rjust(7, "0")
        return os.path.join(
            json_dir, date, order_id[0], order_id[1], order_id[2], order_id + ".json"
        )

    def preprocess_to_json(self) -> None:
        assert self.load
        for date, orders in self.loaded_data.items():
            file_paths = []
            for order in tqdm(orders):
                order_id = order["head"]["order_id"]
                file_path = self.get_json_path(date, order_id)
                file_paths.append(file_path)
                dump_json(order, file_path)
            dump_pickle(file_paths, os.path.join(json_dir, date + ".pickle"))

    def generate_tokenizer(self, tokenizer_dir="."):
        assert self.load
        self.link_tokenizer = categorical_to_index(
            [link["link_id"] for order in self.train_data for link in order["link"]],
            contain_pad=True,
        )
        self.driver_tokenizer = categorical_to_index(
            [order["head"]["driver_id"] for order in self.train_data], contain_pad=True
        )
        self.cross_tokenizer = categorical_to_index(
            [cross["cross_id"] for order in self.train_data for cross in order["cross"]]
        )
        # self.weather_tokenizer = categorical_to_index(self.weather["weather"].values)
        # self.hightemp_tokenizer = categorical_to_index(self.weather["hightemp"].values)
        # self.lowtemp_tokenizer = categorical_to_index(self.weather["lowtemp"].values)
        self.save_tokenizer(tokenizer_dir)

    def save_tokenizer(self, tokenizer_dir="."):
        tokenizers = [
            attr
            for attr in dir(self)
            if attr.endswith("tokenizer") and attr not in self.defined_methods
        ]
        for tokenizer in tokenizers:
            dump_pickle(
                getattr(self, tokenizer),
                os.path.join(tokenizer_dir, tokenizer + ".pickle"),
            )

    def load_tokenizer(self):
        if self.fold == 0:
            tokenizer_dir = self.tokenizer_dir
        else:
            tokenizer_dir = os.path.join(self.tokenizer_dir, "fold%d" % self.fold)

        print("Load tokenizer from %s dir..." % tokenizer_dir)
        for tokenizer in os.listdir(tokenizer_dir):
            if not tokenizer.endswith("tokenizer.pickle"):
                continue
            setattr(
                self,
                tokenizer.replace(".pickle", ""),
                load_pickle(os.path.join(tokenizer_dir, tokenizer)),
            )

    def reindex(self):
        for order in tqdm(self.train_data):
            order["head"]["driver_id"] = self.convert_driver_to_idx(
                order["head"]["driver_id"]
            )
            for link in order["link"]:
                link["link_id"] = self.convert_link_to_idx(link["link_id"])

    def convert_link_to_idx(self, link_id: int):
        assert self.link_tokenizer is not None
        return self.link_tokenizer.encode(link_id)

    def convert_driver_to_idx(self, driver_id: int):
        assert self.driver_tokenizer is not None
        return self.driver_tokenizer.encode(driver_id)

    @property
    def driver_num(self) -> int:
        assert self.driver_tokenizer is not None
        return len(self.driver_tokenizer)

    @property
    def link_num(self) -> int:
        assert self.link_tokenizer is not None
        return len(self.link_tokenizer)

    def extract_dense_data(self, order):
        simple_eta = order["head"]["simple_eta"] / 1000.0
        distance = order["head"]["distance"] / 2000.0
        link_num = len(order["link"])
        cross_num = len(order["cross"])
        approx_speed = (order["head"]["distance"] / order["head"]["simple_eta"]) / 3.6
        data = tensor([simple_eta, distance, link_num, cross_num, approx_speed]).to(
            torch.float
        )
        return data

    def extract_seq_dense_data(self, order):
        link_time = tensor([link["link_time"] / 10.0 for link in order["link"]])
        link_ratio = tensor([link["link_ratio"] for link in order["link"]])
        link_status = [link["link_current_status"] for link in order["link"]]
        link_status = tensor(link_status).to(torch.long)
        link_status = F.one_hot(link_status, 5).to(torch.float)

        dense = torch.stack([link_time, link_ratio], dim=-1)

        # link_gnn_embedding = torch.stack(
        #    [self.node2vec(link["link_id"]).detach() for link in order["link"]]
        # )
        seq_dense = torch.cat([dense, link_status], dim=-1)
        # seq_dense = torch.cat([dense, link_status, link_gnn_embedding], dim=-1)
        return seq_dense

    def extract_seq_sparse_data(self, order):
        link_status = [link["link_current_status"] for link in order["link"]]
        link_status = tensor(link_status, dtype=torch.long)
        # slice_id = [order["head"]["slice_id"] for _ in order["link"]]
        slice_id = [link["link_arrival_slice_id"] for link in order["link"]]
        weekday = [order["head"]["weekday"] for _ in order["link"]]
        weekday = tensor(weekday, dtype=torch.long)
        slice_id = tensor(slice_id, dtype=torch.long)
        seq_sparse = torch.stack([link_status, slice_id, weekday], dim=-1)
        return seq_sparse

    def extract_sparse_data(self, order):
        weekday = order["head"]["weekday"]
        timestamp = order["head"]["slice_id"] // 6

        distance = order["head"]["distance"]
        if distance < 3000:
            distance_class = 0
        elif 3000 <= distance < 7000:
            distance_class = 1
        elif 7000 <= distance < 12000:
            distance_class = 2
        elif 12000 <= distance < 20000:
            distance_class = 3
        else:
            distance_class = 4

        driver = self.driver_tokenizer.encode(order["head"]["driver_id"])
        if self.random_drop:
            if torch.rand(1).item() < 0.005:
                driver = 0

        # weather = self.weather_tokenizer.encode(order["head"]["weather"])
        # hightemp = self.hightemp_tokenizer.encode(order["head"]["hightemp"])
        # lowtemp = self.lowtemp_tokenizer.encode(order["head"]["lowtemp"])

        # sparse = tensor([weekday, timestamp, driver]).to(torch.long)
        sparse = tensor([weekday, timestamp, driver, distance_class], dtype=torch.long)
        return sparse

    def extract_feature(self, order: Dict[str, Any]):
        dense = self.extract_dense_data(order)
        sparse = self.extract_sparse_data(order)

        label = tensor([order["head"]["ata"]]) / 1000.0

        order_id = order["head"]["order_id"]

        link_id = tensor(
            [self.link_tokenizer.encode(link["link_id"]) for link in order["link"]]
        )
        if self.random_drop:
            default_link = torch.zeros_like(link_id)
            link_id = torch.where(
                torch.rand_like(link_id, dtype=torch.float) >= 0.005,
                link_id,
                default_link,
            )
        link_id = link_id.to(torch.long)

        # extract cross id
        cross_id = tensor(
            [
                [
                    self.link_tokenizer.encode(int(cross_link))
                    for cross_link in cross["cross_id"].split("_")
                ]
                for cross in order["cross"]
            ]
        )
        cross_id = torch.cat([torch.tensor([[0, 0]]), cross_id])
        if self.random_drop:
            default_cross = torch.zeros_like(cross_id)
            cross_id = torch.where(
                torch.rand_like(cross_id, dtype=torch.float) >= 0.005,
                cross_id,
                default_cross,
            )
        cross_id = cross_id.to(torch.long)

        if cross_id.ndim != 2:
            print(cross_id)

        cross_id_start = cross_id[:, 0]
        cross_id_end = cross_id[:, 1]
        cross_dense = tensor(
            [[0.0]] + [[cross["cross_time"] / 10.0] for cross in order["cross"]]
        )

        seq_dense = self.extract_seq_dense_data(order)
        seq_sparse = self.extract_seq_sparse_data(order)

        if self.calc_weight is not None:
            weight = tensor([self.calc_weight(order)])
        else:
            weight = tensor([1.0])

        # very short travel time
        if 0 < label < 0.06:
            weight = tensor([0.0])
        # extremely high travel speed
        if (order["head"]["distance"] / order["head"]["ata"]) / 3.6 > 120:
            weight = tensor([0.0])

        seq_label = tensor(
            [[link["link_arrival_status"]] for link in order["link"]], dtype=torch.long
        )

        return (
            dense,
            sparse,
            seq_dense,
            seq_sparse,
            link_id,
            cross_id_start,
            cross_id_end,
            cross_dense,
            seq_label,
            label,
            weight,
            order_id,
        )

    def basic_info(self):
        return {"link_num": self.link_num, "driver_num": self.driver_num}

    def generate_config(self):

        basic_info = self.basic_info()

        deep_config = {
            "dense": {"size": 5},
            "sparse": [
                {"col": 0, "name": "weekday", "size": 7, "dim": 20},
                {"col": 1, "name": "slice_id", "size": 48, "dim": 20},
                {"col": 2, "name": "driver_id", "size": self.driver_num, "dim": 64},
                {"col": 3, "name": "distance", "size": 5, "dim": 20}
                # {"col": 3, "name": "weather", "size": 5, "dim": 20},
                # {"col": 4, "name": "hightemp", "size": 7, "dim": 20},
                # {"col": 5, "name": "lowtemp", "size": 5, "dim": 20},
            ],
        }

        wide_config = {
            "dense": {"size": 5},
            "sparse": [
                {"col": 0, "name": "weekday", "size": 7, "dim": 20},
                {"col": 1, "name": "slice_id", "size": 48, "dim": 20},
                {"col": 3, "name": "distance", "size": 5, "dim": 20}
                # {"col": 0, "name": "link_id", "size": self.link_num, "dim": 20}
                # {"col": 3, "name": "weather", "size": 5, "dim": 20},
                # {"col": 4, "name": "hightemp", "size": 7, "dim": 20},
                # {"col": 5, "name": "lowtemp", "size": 5, "dim": 20},
            ],
        }

        rnn_config = {
            "dense": {"size": 7},
            "sparse": [
                {"col": 0, "name": "link_status", "size": 5, "dim": 20},
                {"col": 1, "name": "slice_id", "size": 288, "dim": 20},
                {"col": 2, "name": "weekday", "size": 7, "dim": 20},
            ],
        }

        return basic_info, wide_config, deep_config, rnn_config


def collate_fn(batch):
    (
        dense,
        sparse,
        seq_dense,
        seq_sparse,
        link_id,
        cross_id_start,
        cross_id_end,
        cross_dense,
        seq_label,
        label,
        weight,
        order_id,
    ) = list(zip(*batch))

    dense = torch.stack(dense)
    sparse = torch.stack(sparse)
    label = torch.stack(label)
    weight = torch.stack(weight)

    link_len = list(map(len, link_id))
    link_id = pad_sequence(link_id, batch_first=True)
    seq_dense = pad_sequence(seq_dense, batch_first=True)
    seq_sparse = pad_sequence(seq_sparse, batch_first=True)
    seq_label = pad_sequence(seq_label, batch_first=True)

    cross_len = list(map(len, cross_id_start))
    cross_id_start = pad_sequence(cross_id_start, batch_first=True)
    cross_id_end = pad_sequence(cross_id_end, batch_first=True)
    cross_dense = pad_sequence(cross_dense, batch_first=True)

    return (
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
    )


if __name__ == "__main__":
    # dataset = GISCUPDataset("test", load=True, flush=True)
    # dataset.preprocess_to_json()
    dataset = GISCUPDataset("test", load=True)
    dataset.preprocess_to_json()
    exit(0)
    pprint(dataset.train_data[:10])
    # df = pd.DataFrame(pd.json_normalize(dataset.train_data[:10]))
    # print(df.columns)

    print("*" * 20)
    exit(0)

    dataset.generate_tokenizer("tokenizer")
    print(dataset[0])
    print("*" * 20)
    exit(0)

    dataset = GISCUPDataset("train_val", load=True)
    # extract feature in preprocess_to_json method
    dataset.preprocess_to_json()
