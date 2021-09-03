import argparse
import os
from pprint import pprint
from typing import Any, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from dataset import GISCUPDataset, Tokenizer, collate_fn
from plmodel import GISCUPModel


def parse_args():
    parser = argparse.ArgumentParser("Test")

    parser.add_argument(
        "--data_dir", type=str, default="/data3/ganyunchong/giscup_2021"
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, default="/nvme/ganyunchong/didi/10fold"
    )
    parser.add_argument(
        "--kfold_data_dir", type=str, default="/nvme/ganyunchong/didi/10fold"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--load", action="store_true")

    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--train_end", type=str, default="20200824")
    parser.add_argument("--val_end", type=str, default="20200831")

    parser.add_argument("--cross", dest="cross", action="store_true")
    parser.add_argument("--no-cross", dest="cross", action="store_false")
    parser.set_defaults(cross=True)

    parser.add_argument("--aux-loss", dest="aux_loss", action="store_true")
    parser.add_argument("--no-aux-loss", dest="aux_loss", action="store_false")
    parser.set_defaults(aux_loss=True)

    return parser.parse_args()


def test(args):
    pl.seed_everything(args.seed)

    cpu_num = os.cpu_count()
    assert isinstance(cpu_num, int)

    train_dataset = GISCUPDataset(
        dataset_type="train",
        train_end=args.train_end,
        validation_end=args.val_end,
        fold=args.fold,
        tokenizer_dir=args.tokenizer_dir,
        kfold_data_dir=args.kfold_data_dir,
    )
    val_dataset = GISCUPDataset(
        dataset_type="val",
        train_end=args.train_end,
        validation_end=args.val_end,
        fold=args.fold,
        tokenizer_dir=args.tokenizer_dir,
        kfold_data_dir=args.kfold_data_dir,
    )
    test_dataset = GISCUPDataset(
        dataset_type="test",
        fold=args.fold,
        tokenizer_dir=args.tokenizer_dir,
        kfold_data_dir=args.kfold_data_dir,
    )

    train_dataset.load_tokenizer()
    val_dataset.load_tokenizer()
    test_dataset.load_tokenizer()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    if args.fold != 0:
        submission_file = "submission_fold%d.csv" % args.fold
    else:
        submission_file = "submission.csv"

    model = GISCUPModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
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
    args = parse_args()
    test(args)
