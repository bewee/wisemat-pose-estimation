import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import os

FOLDER = "SLP"
# FOLDER = "bodies-at-rest"

FILE_IN = os.path.join(FOLDER, "full-data.pkl")
FILE_TRAIN_OUT = os.path.join(FOLDER, "data-train.pkl")
FILE_VAL_OUT = os.path.join(FOLDER, "data-val.pkl")
FILE_TEST_OUT = os.path.join(FOLDER, "data-test.pkl")
PORTION_TRAIN = 0.8
PORTION_VAL = 0.1
PORTION_TEST = 1.0 - (PORTION_TRAIN + PORTION_VAL)

df = pd.read_pickle(FILE_IN)

if not('participant' in df):
    df['participant'] = range(len(df))

unique_participants = np.unique(df['participant'])
permutation = np.random.permutation(unique_participants)

train_set_size = int(len(unique_participants) * PORTION_TRAIN)
val_set_size = int(len(unique_participants) * PORTION_VAL)
                     
train_set_participants = permutation[:train_set_size]
val_set_participants = permutation[train_set_size:train_set_size+val_set_size]
test_set_participants = permutation[train_set_size+val_set_size:]
                     
train_set = df[df['participant'].isin(train_set_participants)]
train_set = train_set.drop(columns=['participant'])
train_set.reset_index(drop=True, inplace=True)
val_set = df[df['participant'].isin(val_set_participants)]
val_set = val_set.drop(columns=['participant'])
val_set.reset_index(drop=True, inplace=True)
test_set = df[df['participant'].isin(test_set_participants)]
test_set = test_set.drop(columns=['participant'])
test_set.reset_index(drop=True, inplace=True)

with pd.option_context('display.max_rows',10):
    print("train set")
    print(train_set)
    print("val set")
    print(val_set)
    print("test set")
    print(test_set)

train_set.to_pickle(FILE_TRAIN_OUT)
val_set.to_pickle(FILE_VAL_OUT)
test_set.to_pickle(FILE_TEST_OUT)
