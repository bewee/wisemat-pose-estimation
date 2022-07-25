import torch
import numpy as np
from torchvision.transforms import ToTensor
import cv2
from skimage.filters import sobel
from .constants import constants

class Stack(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        transformed_sample = [transform(sample) for transform in self.transforms]
        return torch.cat(transformed_sample, axis=0)

class Boolean(object):
    def __call__(self, sample):
        return (sample > 0).type(torch.float32)

class Identity(object):
    def __call__(self, sample):
        return sample.type(torch.float32) / 255
    
class MinMaxNormalize(object):
    def __call__(self, sample):
        return (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))

class StdMeanNormalize(object):
    def __call__(self, sample):
        return (sample - torch.mean(sample)) / torch.std(sample)

class HistogramEqualize(object):
    def __call__(self, sample):
        sample_shape = sample.shape
        sample = sample.numpy().astype(np.uint8)
        if len(sample.shape) == 3:
            sample = sample.squeeze(axis=0)
        return ToTensor()(cv2.equalizeHist(sample))

class SobelFilter(object):
    def __call__(self, sample):
        sample_shape = sample.shape
        sample = sample.numpy().astype(np.uint8)
        if len(sample.shape) == 3:
            sample = sample.squeeze(axis=0)
        return ToTensor()(sobel(sample)).type(torch.FloatTensor)

class SkeletonToHeatmaps(object):
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, sample):
        def heatmap_for_position(px, py):
            tx = np.arange(0, constants.SENSORS_Y, dtype=np.float32) - (1-py) * (constants.SENSORS_Y-1)
            tx = np.repeat(tx, constants.SENSORS_X).reshape(constants.SENSORS_Y, constants.SENSORS_X)
            tx = np.square(tx)

            ty = np.arange(0, constants.SENSORS_X, dtype=np.float32) - px * (constants.SENSORS_X-1)
            ty = np.transpose(np.repeat(ty, constants.SENSORS_Y).reshape(constants.SENSORS_X, constants.SENSORS_Y))
            ty = np.square(ty)

            t = np.exp(-(tx + ty) / (2 * self.sigma ** 2))
            return t

        sample = sample.reshape(-1, 2)
        heatmaps = np.asarray([heatmap_for_position(px.item(), py.item()) for [px, py] in sample])
        return heatmaps

class HeatmapsToSkeleton(object):
    def __call__(self, sample):
        skeleton = []
        for i in range(sample.shape[0]):
            skeleton += [np.unravel_index(sample[i].argmax(), sample[i].shape)]

        skeleton = np.asarray(skeleton).reshape(-1, 2).astype(np.float32)
        skeleton[:,0] = 1 - skeleton[:,0] / (constants.SENSORS_Y-1)
        skeleton[:,1] = skeleton[:,1] / (constants.SENSORS_X-1)

        tmp = np.copy(skeleton[:,0])
        skeleton[:,0] = skeleton[:,1]
        skeleton[:,1] = tmp
        return skeleton

class SkeletonCleverToCommon(object):
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = sample.reshape(24, -1)
        sample = np.asarray([
            sample[8],
            sample[5],
            sample[2],
            sample[1],
            sample[4],
            sample[7],
            sample[21],
            sample[19],
            (sample[14] + sample[17]) / 2,
            (sample[13] + sample[16]) / 2,
            sample[18],
            sample[22],
            sample[12],
        ])
        sample = sample[:,[0,1]]
        return sample

class SkeletonSLPToCommon(object):
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = sample.reshape(14, -1)
        return sample[:13,[0,1]]

class SkeletonCommonToArmsOmitted(object):
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = sample.reshape(13, -1)
        return np.concatenate((sample[:6,:], sample[8:10,:], sample[12:13,:]), axis=0)

class HeatmapsCommonToArmsOmitted(object):
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = sample.reshape(13, constants.SENSORS_Y, constants.SENSORS_X)
        return np.concatenate((sample[:6,:,:], sample[8:10,:,:], sample[12:13,:,:]), axis=0)
