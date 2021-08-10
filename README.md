# SIGSPATIAL-2021-GISCUP-3rd-Solution

**Competition Page**: https://www.biendata.xyz/competition/didi-eta/

## Prepare Data

Download the dataset from [here](https://www.biendata.xyz/competition/didi-eta/data/) and change `data_dir` in `dataset.py`.

```
python dataset.py
```

It will preprocess the original `.txt` files, convert them into `.json` files and `.pickle` files to accelerate the data loading.

Then it will split the whole train dataset into 5Fold and 10Fold.

## Train & Test

### Train

```
python train.py
```

### Test

```
python test.py
```

## Data Ensemble

Use the simple average result to generate the final submission.

The final leaderboard result is the average of 5fold and 10fold (15 model in total).

```
python merge_submission.py
```
