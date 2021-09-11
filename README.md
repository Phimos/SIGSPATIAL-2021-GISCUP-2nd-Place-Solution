# SIGSPATIAL-2021-GISCU-2nd-Place-Solution

**Competition Page:** [DiDi-ETA](https://www.biendata.xyz/competition/didi-eta/)

**Final Official Ranking**

<table>
    <tr>
        <th>Ranking</th>
        <th>Award (Cash Prize)</th>
        <th>Name</th>
        <th>MAPE</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Champion ($10,000)</td>
        <td>单模CBT</td>
        <td>0.11974</td>
    </tr>
    <tr>
        <td>2</td>
        <td rowspan="2">Runner-ups</br>
($5,000 each team)</td>
        <td>Pims</td>
        <td>0.12099</td>
    </tr>
    <tr>
        <td>3</td>
        <td>华南工农联盟</td>
        <td>0.12116</td>
    </tr>
    <tr>
        <td>4</td>
        <td rowspan="2">Second Runner-ups</br>
($2,500 each team)</td>
        <td>机器算命</td>
        <td>0.12177</td>
    </tr>
    <tr>
        <td>5</td>
        <td>pumbaa</td>
        <td>0.12198</td>
    </tr>
    <tr>
        <td>6</td>
        <td rowspan="2">Recognition Award</br>
($1,000 each team)</td>
        <td>MobiLab</td>
        <td>0.12478</td>
    </tr>
    <tr>
        <td>7</td>
        <td>悦智AI实验室</td>
        <td>0.12511</td>
    </tr>
</table>

**Team Name:** Pims

**Team Members:** Yunchong Gan, Mingjie Wang, Haoyu Zhang

## Quick Start

### Prepare Data

Download the dataset from [here](https://www.biendata.xyz/competition/didi-eta/data/) and change `data_dir` in `dataset.py`.

```
python dataset.py
```

It will preprocess the original `.txt` files, convert them into `.json` files and `.pickle` files to accelerate the data loading.

Then it will split the whole train dataset into 5Fold and 10Fold.

### Train & Test

#### Train

```
python train.py
```

#### Test

```
python test.py
```

### Data Ensemble

Use the simple average result to generate the final submission.

The final leaderboard result is the average of 5fold and 10fold (15 model in total).

```
python merge_submission.py
```

## Details

### Model Architecture

The whole model based on WDR, Didi ETA paper in KDD2018.

```
Wide \
      \
Deep --- concat - MLP - Prediction
      /
RNN -/
 |
 |----Predict Current Link Status
```

### Input

**Wide**

| Name                  | Type        | Number of Embedding | Embedding Dim | Description |
| --------------------- | ----------- | ------------------- | ------------- | ----------- |
| Simple ETA            | Numeric     |                     | 1             |             |
| Distance              | Numeric     |                     | 1             |             |
| Link Number           | Numeric     |                     | 1             |             |
| Cross Number          | Numeric     |                     | 1             |             |
| Approximate Speed     | Numeric     |                     | 1             |             |
| Weekday               | Categorical | 7                   | 1             |             |
| Slice ID              | Categorical | 48                  | 1             |             |
| Distance(Categorical) | Categorical | 5                   | 1             |             |

**Deep**

| Name                  | Type        | Number of Embedding | Embedding Dim | Description          |
| --------------------- | ----------- | ------------------- | ------------- | -------------------- |
| Simple ETA            | Numeric     |                     | 1             |                      |
| Distance              | Numeric     |                     | 1             |                      |
| Link Number           | Numeric     |                     | 1             |                      |
| Cross Number          | Numeric     |                     | 1             |                      |
| Approximate Speed     | Numeric     |                     | 1             |                      |
| Weekday               | Categorical | 7                   | 20            |                      |
| Slice ID              | Categorical | 48                  | 20            |                      |
| Driver ID             | Categorical | depend on dataset   | 64            |                      |
| Distance(Categorical) | Categorical | 5                   | 20            | Split in 3/7/12/20km |

**RNN - Link**

| Name                | Type        | Number of Embedding | Embedding Dim | Description                               |
| ------------------- | ----------- | ------------------- | ------------- | ----------------------------------------- |
| Link Time           | Numeric     |                     | 1             |                                           |
| Link Ratio          | Numeric     |                     | 1             |                                           |
| Link Status(Onehot) | Numeric     |                     | 5             |                                           |
| Weekday             | Categorical | 7                   | 20            |                                           |
| Slice ID            | Categorical | 288                 | 20            | compute with slice id and link/cross time |
| Link ID             | Categorical | depend on dataset   | 20            |                                           |

**RNN - Cross**
| Name          | Type        | Number of Embedding | Embedding Dim | Description |
| ------------- | ----------- | ------------------- | ------------- | ----------- |
| Cross Time    | Numeric     |                     | 1             |             |
| Start Link ID | Categorical | depend on dataset   | 20            |             |
| End Link ID   | Categorical | depend on dataset   | 20            |             |


## Different from WDR
* Auxiliary Loss for Link Status Classification
* Concat result from different branches
* Random Split KFold
* Model Ensemble
