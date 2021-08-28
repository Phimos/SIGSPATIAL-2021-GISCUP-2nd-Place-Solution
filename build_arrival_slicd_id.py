import glob

import pandas as pd
from tqdm import tqdm

from dataset import dump_json, load_json, load_pickle

for order_path in tqdm(
    glob.iglob("/data3/ganyunchong/giscup_2021/json/**/*.json", recursive=True)
):
    order = load_json(order_path)

    date = int(order["head"]["date"])
    slice_id = order["head"]["slice_id"]
    weekday = order["head"]["weekday"]
    sum_time = 0
    crosses = order["cross"]
    cross_idx = 0

    for link in order["link"]:
        if (
            cross_idx != len(crosses)
            and str(link["link_id"]) in crosses[cross_idx]["cross_id"]
        ):
            sum_time += crosses[cross_idx]["cross_time"]
            cross_idx += 1
        sum_time += link["link_time"]
        arrival_slice_id = (slice_id + round(sum_time / 300)) % 288
        link["link_arrival_slice_id"] = arrival_slice_id

    dump_json(order, order_path)
