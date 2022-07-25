import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .constants import constants

class PressurePoseDataset(Dataset):
    def __init__(
        self,
        index_file,
        image_dir,
        input_format='inplace',
        target_format='coordinates',
        input_transform=None,
        target_transform=None,
        preprocess_input_transform=True,
        preprocess_target_transform=True
    ):
        self.index = pd.read_pickle(index_file)
        self.image_dir = image_dir
        self.input_format = input_format
        self.target_format = target_format
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.preprocess_input_transform = preprocess_input_transform
        self.preprocess_target_transform = preprocess_target_transform
        
        if self.input_format == 'inplace':
            self.index['pressure_image'] = [
                self._input_tensor(entry[1]['pressure_image'])
                for entry in self.index.iterrows()
            ]
        if self.input_transform and self.input_format == 'inplace' and self.preprocess_input_transform:
            self.index['pressure_image'] = [
                self.input_transform(entry[1]['pressure_image'])
                for entry in self.index.iterrows()
            ]
        if self.target_format == 'coordinates':
            self.index['skeleton'] = [
                self._target_tensor(entry[1]['skeleton'])
                for entry in self.index.iterrows()
            ]
        if self.target_transform and self.target_format == 'coordinates' and self.preprocess_target_transform:
            self.index['skeleton'] = [
                self.target_transform(entry[1]['skeleton'])
                for entry in self.index.iterrows()
            ]

    def _input_tensor(self, x):
        x = x.reshape(1, constants.SENSORS_Y, constants.SENSORS_X).astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def _target_tensor(self, y):
        if self.target_format == 'coordinates':
            y = y[:,[0,1]].flatten()
        y = torch.from_numpy(y)
        return y
                
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index.iloc[idx]

        x = None
        if self.input_format == 'inplace':
            x = item['pressure_image']
        elif self.input_format == 'ref':
            img_path = os.path.join(self.image_dir, f"{item['pressure_image_ref']}.npy")
            x = np.load(img_path)
            x = self._input_tensor(x)
        else:
            raise Exception("Invalid input format")
        if self.input_transform and not(self.input_format == 'inplace' and self.preprocess_input_transform):
            x = self.input_transform(x)

        y = None
        if self.target_format == 'coordinates':
            y = item["skeleton"]
        elif self.target_format == 'heatmaps':
            heatmap_path = os.path.join(self.image_dir, f"{item['pressure_image_ref']}_hm.npy")
            y = np.load(heatmap_path)
            y = self._target_tensor(y)
        else:
            raise Exception("Invalid target format")
        if self.target_transform and not(self.target_format == 'coordinates' and self.preprocess_target_transform):
            y = self.target_transform(y)

        return x, y
