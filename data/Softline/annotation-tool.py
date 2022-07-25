import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from lib.viz import plot_mattress,plot_skeleton
from lib.transforms import Identity, StdMeanNormalize, Boolean, HistogramEqualize, SobelFilter
import math
import pandas as pd
import numpy as np
import torch
import os

def dist(x, y, x_, y_):
    return math.sqrt( (x-x_)**2 + (y-y_)**2 )

class Index:
    views = [Identity(), Boolean(), HistogramEqualize(), SobelFilter()]
    initial_skeleton = np.array([
        [0.25, 0.90],
        [0.25, 0.70],
        [0.25, 0.50],
        [0.75, 0.50],
        [0.75, 0.70],
        [0.75, 0.90],
        [0.10, 0.50],
        [0.15, 0.35],
        [0.25, 0.20],
        [0.75, 0.20],
        [0.85, 0.35],
        [0.90, 0.50],
        [0.50, 0.10],
    ])

    def __init__(self, ax):
        self.df = pd.read_pickle('tool-index.p')
        self.ax = ax
        self.view = 0
        self.skeleton = self.initial_skeleton.copy()
        self.dragindex = None

        self.next_image(None)
        
        canvas = ax.figure.canvas
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
    def on_button_press(self, event):
        if event.xdata == None or event.ydata == None:
            return 
        eps = 1
        d_min = float("inf")
        target = None
        for (i, point) in enumerate(self.skeleton):
            d = dist(event.xdata, event.ydata, point[0]*27, point[1]*64)
            if d <= eps and d < d_min:
                target = i
                d_min = d
        if target != None:
            self.dragindex = target

    def on_mouse_move(self, event):
        if self.dragindex == None:
            return
        x_rel = event.xdata / 27
        y_rel = event.ydata / 64
        self.skeleton[self.dragindex][0] = x_rel
        self.skeleton[self.dragindex][1] = y_rel
        self.render()

    def on_button_release(self, event):
        self.on_mouse_move(event)
        self.dragindex = None

    def next_image(self, event):
        self.skeleton = self.initial_skeleton.copy()
        self.entry = self.df.sample().iloc[0]
        data = np.genfromtxt(f"Frames_csv/{self.entry['measurement']}/{self.entry['frame']}", delimiter=',')
        image = np.zeros((64, 27))
        image[:,:-1] = data.T
        self.image = torch.Tensor(image)
        self.render()
    
    def next_view(self, event):
        self.view = (self.view + 1) % len(self.views)
        self.render()
        
    def render(self):
        self.ax.clear()
        plot_mattress(self.views[self.view](self.image.flip(dims=(0,))), ax=self.ax, invert_y=False, cbar=False, xticklabels=False, yticklabels=False)
        plot_skeleton(self.skeleton, col='green', ax=self.ax, show_z=False, show_labels=True)
        self.ax.set_title(f"{self.entry['measurement']} / {self.entry['frame']}")
        self.ax.figure.canvas.draw()

    def save(self, event):
        df = None
        if os.path.exists('full-data_inplace.pkl'):
            df = pd.read_pickle('full-data_inplace.pkl')
        else:
            df = pd.DataFrame(columns = ['pressure_image', 'skeleton', 'measurement', 'frame'])
        self.skeleton[:,1] = 1 - self.skeleton[:,1]
        df.loc[len(df)] = [self.image.numpy(), np.asarray(self.skeleton), self.entry['measurement'], self.entry['frame']]
        df.to_pickle('full-data_inplace.pkl')
        self.next_image(event)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    index = Index(ax)

    axnext = plt.axes([0.45, 0.05, 0.15, 0.05])
    bnext = Button(axnext, 'Next Image')
    bnext.on_clicked(index.next_image)

    axview = plt.axes([0.25, 0.05, 0.15, 0.05])
    bview = Button(axview, 'Toggle View')
    bview.on_clicked(index.next_view)
    
    axsave = plt.axes([0.05, 0.05, 0.15, 0.05])
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(index.save)
    
    plt.show()
