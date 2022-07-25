import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from glob import glob
import zipfile
from alive_progress import alive_bar
from lib.constants import constants, set_format
from create_dataset_lib import CreateDatasetLib

set_format('hrl-ros')

df = pd.DataFrame(columns=['pressure_image', 'skeleton', 'participant', 'male', 'height', 'mass'])

count = 0
for zfile in glob("data/*.zip"):
    z = zipfile.ZipFile(zfile) 
    for pfile in z.namelist():
        if pfile.endswith('.p'):
            count += 1

[p_world_mat, R_world_mat] = pd.read_pickle('data/mat_axes.p')

with alive_bar(count) as bar:
    participant = 0
    for zfile in glob("data/*.zip"):
        z = zipfile.ZipFile(zfile) 
        for pfile in z.namelist():
            if pfile.endswith('.p'):
                bar()
                try:
                    data = pd.read_pickle(z.open(pfile))
                except:
                    print('Errornous pickle detected')
                for sample in range(len(data)):
                    p_mat_raw = np.asarray(data[sample][0]).reshape(constants.SENSORS_Y, constants.SENSORS_X)
                    p_mat_raw = (p_mat_raw.astype(np.float32) / 100 * 255).astype(np.uint8)
                    target_raw = data[sample][1]
                    target_mat = CreateDatasetLib().world_to_mat(target_raw, p_world_mat, R_world_mat)
                    target_mat[:,0] -= constants.MATTRESS_WIDTH/3
                    target_mat[:,0] /= constants.MATTRESS_WIDTH
                    target_mat[:,1] -= constants.MATTRESS_HEIGHT/6
                    target_mat[:,1] /= constants.MATTRESS_HEIGHT

                    df.loc[len(df)] = [p_mat_raw, target_mat, participant, None, None, None]
        participant += 1

with pd.option_context('display.max_rows',10):
    print(df)

df.sample(frac=1).to_pickle(f"full-data_inplace.pkl")
