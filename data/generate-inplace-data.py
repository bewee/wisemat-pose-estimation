import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from alive_progress import alive_bar
import os

FOLDER = "."
# FOLDER = "SLP"
# FOLDER = "bodies-at-rest"

FILE = "data-train"
# FILE = "data-val"
# FILE = "data-test"

FILE_IN = os.path.join(FOLDER, f"{FILE}.pkl")
FILE_OUT = os.path.join(FOLDER, f"{FILE}_inplace.pkl")

df = pd.read_pickle(FILE_IN)

pressure_images = []

with alive_bar(len(df)) as bar:
    for entry in df.iterrows():
        path = os.path.join(FOLDER, f"{entry[1]['pressure_image_ref']}.npy")
        image = np.load(path)
        pressure_images += [image]
        bar()

df["pressure_image"] = pressure_images

with pd.option_context('display.max_rows',10):
    print(df)

df.to_pickle(FILE_OUT)
