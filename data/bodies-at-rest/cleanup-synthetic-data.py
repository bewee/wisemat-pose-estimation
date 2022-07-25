import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from smpl_webuser.serialization import load_model
from alive_progress import alive_bar
from lib.constants import constants

ORIGIN = 'synth'
CATEGORY = 'general'
GENDER = 'f'
ID = f"test_rollpi_plo_{GENDER}_lay_set23to24_3000"
FILE_NAME = f"{ORIGIN}/{CATEGORY}/{ID}"
MODEL_NAME = f"models/basicModel_{GENDER}_lbs_10_207_0_v1.0.0.pkl"

def calculate_skeleton(i):
    model = load_model(MODEL_NAME)
    model.pose[:] = data['joint_angles'][i]
    model.betas[:] = data['body_shape'][i]
    
    joint_cart_gt = np.array(model.J_transformed).reshape(24, 3)
    for s in range(3):
        joint_cart_gt[:, s] += (data['root_xyz_shift'][i][s] - float(model.J_transformed[0, s]))
        
    joint_cart_gt = joint_cart_gt / np.asarray([constants.MATTRESS_WIDTH, constants.MATTRESS_HEIGHT, 1])
    joint_cart_gt = joint_cart_gt - np.asarray([0.35, 0.15, 0]) # I don't know why, but all skeletons seem to be shifted by about this amount ¯\_(ツ)_/¯
    
    return joint_cart_gt

data = pd.read_pickle(f"{FILE_NAME}.p")
df = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'male', 'height', 'mass'])

with alive_bar(len(data['images'])) as bar:
    for i in range(len(data['images'])):
        skeleton = calculate_skeleton(i)
        
        im = data['images'][i].reshape((constants.SENSORS_Y, constants.SENSORS_X))
        im_path = f"refs/{ORIGIN}_{CATEGORY}_{ID}_{i}"
        np.save(im_path, im.astype(np.uint8))
        
        df.loc[len(df)] = [
            im_path,
            skeleton,
            GENDER == 'm',
            data['body_height'][i]*100,
            data['body_mass'][i],
        ]

        bar()
        
with pd.option_context('display.max_rows',10):
    print(df)

df.to_pickle(f"{FILE_NAME}.pkl")
