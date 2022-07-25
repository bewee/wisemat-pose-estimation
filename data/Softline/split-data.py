import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import os

FILE_IN = "full-data_inplace.pkl"
FILE_VAL_OUT = "data-val_inplace.pkl"
FILE_TEST_OUT = "data-test_inplace.pkl"
PORTION_VAL = 0.5

df = pd.read_pickle(FILE_IN)
                     
val_set = df.sample(frac=PORTION_VAL)
val_set.reset_index(drop=True, inplace=True)

test_set = pd.concat([df, val_set]).drop_duplicates(subset=['measurement', 'frame'], keep=False)
test_set.reset_index(drop=True, inplace=True)

with pd.option_context('display.max_rows',10):
    print("val set")
    print(val_set)
    print("test set")
    print(test_set)

val_set.to_pickle(FILE_VAL_OUT)
test_set.to_pickle(FILE_TEST_OUT)
