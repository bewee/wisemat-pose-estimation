import sys
import os
import importlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lib.datamodule import PressurePoseDataModule
from lib.transforms import *
from lib.F import mpjpe, per_joint_position_errors, pcp, pck, mpjpe_arms_omitted

class Baseline:
    def __init__(self):
        dm = PressurePoseDataModule(input_transform=Stack([Identity()]))
        dm.setup()
        train_data = next(iter(DataLoader(dm.train_set, batch_size=len(dm.train_set))))
        self.skel = torch.mean(train_data[1], 0).reshape(-1, 2)
        
    def __call__(self, arg):
        return self.skel.repeat(arg.shape[0], 1, 1)
