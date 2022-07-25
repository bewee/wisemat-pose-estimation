import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from alive_progress import alive_bar
from lib.transforms import SkeletonToHeatmaps

FILE = 'data-train.pkl'
# FILE = 'data-val.pkl'
# FILE = 'data-test.pkl'

df = pd.read_pickle(FILE)

with alive_bar(len(df)) as bar:
    for entry in df.iterrows():
        path = entry[1]['pressure_image']
        skeleton = entry[1]['skeleton']
        skeleton = skeleton.astype(np.float32)
        skeleton = skeleton[:,[0,1]]
        heatmaps = SkeletonToHeatmaps()(skeleton)
        hm_path = f"{path}_hm"
        np.save(hm_path, heatmaps)
        bar()
