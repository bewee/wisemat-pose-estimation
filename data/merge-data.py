import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from lib.transforms import SkeletonCleverToCommon, SkeletonSLPToCommon

def append_bodies_at_rest():
    global df_train, df_test, df_val

    df_bodies_at_rest_train = pd.read_pickle('bodies-at-rest/data-train.pkl')
    df_bodies_at_rest_train['skeleton'] = df_bodies_at_rest_train['skeleton'].map(SkeletonCleverToCommon())
    df_bodies_at_rest_train['pressure_image_ref'] = df_bodies_at_rest_train['pressure_image_ref'].map(lambda x: f"bodies-at-rest/{x}")

    df_train = pd.concat([df_train, df_bodies_at_rest_train], ignore_index=True)

def append_slp():
    global df_train, df_test, df_val

    df_slp_train = pd.read_pickle('SLP/data-train.pkl')
    df_slp_train['skeleton'] = df_slp_train['skeleton'].map(SkeletonSLPToCommon())
    df_slp_train['pressure_image_ref'] = df_slp_train['pressure_image_ref'].map(lambda x: f"SLP/{x}")

    df_slp_val = pd.read_pickle('SLP/data-val.pkl')
    df_slp_val['skeleton'] = df_slp_val['skeleton'].map(SkeletonSLPToCommon())
    df_slp_val['pressure_image_ref'] = df_slp_val['pressure_image_ref'].map(lambda x: f"SLP/{x}")

    df_slp_test = pd.read_pickle('SLP/data-test.pkl')
    df_slp_test['skeleton'] = df_slp_test['skeleton'].map(SkeletonSLPToCommon())
    df_slp_test['pressure_image_ref'] = df_slp_test['pressure_image_ref'].map(lambda x: f"SLP/{x}")

    oversample_slp_factor = 9
    for i in range(oversample_slp_factor):
        df_train = pd.concat([df_train, df_slp_train], ignore_index=True)

    df_val = pd.concat([df_val, df_slp_val], ignore_index=True)
    df_test = pd.concat([df_test, df_slp_test], ignore_index=True)

def shuffle_datasets():
    global df_train, df_test, df_val

    df_train = df_train.sample(frac=1)
    df_train.reset_index(drop=True, inplace=True)
    df_val = df_val.sample(frac=1)
    df_val.reset_index(drop=True, inplace=True)
    df_test = df_test.sample(frac=1)
    df_test.reset_index(drop=True, inplace=True)

df_train = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'male', 'height', 'mass'])
df_test = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'male', 'height', 'mass'])
df_val = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'male', 'height', 'mass'])

append_bodies_at_rest()
append_slp()
shuffle_datasets()

with pd.option_context('display.max_rows',10):
    print("train set")
    print(df_train)
    print("val set")
    print(df_val)
    print("test set")
    print(df_test)

df_train.to_pickle('data-train.pkl')
df_val.to_pickle('data-val.pkl')
df_test.to_pickle('data-test.pkl')
