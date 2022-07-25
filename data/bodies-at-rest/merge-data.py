import pandas as pd
import glob

FILES = [
    'synth/general/*.pkl'
]
FILE_OUT = 'full-data.pkl'

df = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'male', 'height', 'mass'])

for glob_str in FILES:
    for file in glob.glob(glob_str):
        df = pd.concat([df, pd.read_pickle(file)], ignore_index=True)

with pd.option_context('display.max_rows',10):
    print(df)

df.to_pickle(FILE_OUT)
