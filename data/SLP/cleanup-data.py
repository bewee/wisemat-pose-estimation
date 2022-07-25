import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import scipy.io
import cv2
import os
from alive_progress import alive_bar
from lib.constants import constants

def convert(lab, participant, cover, sample):
    pm = np.load(f"{lab}/{participant}/PMarray/{cover}/{sample}.npy")
    pm_r = cv2.resize(pm, (constants.SENSORS_X, constants.SENSORS_Y))
    pm_r = np.asarray(pm_r).astype(np.uint8)
    pm_path = f"refs/{lab}_{participant}_{cover}_{sample}"
    np.save(pm_path, pm_r)
    
    physique = np.load(f"{lab}/physiqueData.npy")[int(sample)-1]
    
    skeletons = np.swapaxes(scipy.io.loadmat(f"{lab}/{participant}/joints_gt_RGB.mat")['joints_gt'], 0, 2)
    skeleton = skeletons[int(sample)-1][:,0:2]
    H_RGB = np.load(f"{lab}/{participant}/align_PTr_RGB.npy")
    skeleton_n = np.pad(skeleton, [(0, 0), (0, 1)], mode='constant', constant_values = (1))
    skeleton_n = np.matmul(skeleton_n, np.swapaxes(H_RGB, 0, 1))
    skeleton_n[:,0] /= skeleton_n[:,2]
    skeleton_n[:,1] /= skeleton_n[:,2]
    skeleton_n = skeleton_n[:,0:2]
    skeleton_n[:,0] /= pm.shape[1]
    skeleton_n[:,1] /= pm.shape[0]
    skeleton_n[:,1] = 1 - skeleton_n[:,1]
    
    df.loc[len(df)] = [pm_path, skeleton_n, int(participant), bool(physique[0]), physique[1], physique[2]]

df = pd.DataFrame(columns=['pressure_image_ref', 'skeleton', 'participant', 'male', 'height', 'mass'])

LABS = ['danaLab']
COVERS = ['uncover', 'cover1', 'cover2']

count = 0
for lab in LABS:
    for participant in next(os.walk(lab))[1]:
        for cover in COVERS:
            for sample_file in next(os.walk(f"{lab}/{participant}/RGB/{cover}"))[2]:
                count += 1

with alive_bar(count) as bar:
    for lab in LABS:
        for participant in next(os.walk(lab))[1]:
            for cover in COVERS:
                for sample_file in next(os.walk(f"{lab}/{participant}/RGB/{cover}"))[2]:
                    sample = sample_file[6:12]
                    convert(lab, participant, cover, sample)
                    bar()

with pd.option_context('display.max_rows',10):
    print(df)

df.to_pickle(f"full-data.pkl")
