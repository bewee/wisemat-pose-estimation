import os
import pandas as pd
from glob import glob

df = pd.DataFrame(columns=['measurement', 'frame'])

for measurement in next(os.walk("Frames_csv"))[1]:
    print(measurement)
    for frame in glob(f"Frames_csv/{measurement}/*.csv"):
        df.loc[len(df)] = [measurement, os.path.basename(frame)]
        
df.to_pickle('tool-index.p')
	