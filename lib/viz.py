import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn
import colorsys
import numpy as np
from .constants import constants

def saturate_color(rgb, l_scale):
    h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    return colorsys.hls_to_rgb(h, min(l * l_scale, 1), s=s)

def plot_line(a, b, col='cyan', ax=plt):
    ax.plot([a[0],b[0]], [a[1],b[1]], color=col)

def plot_skeleton(points, ax=None, col='cyan',
	    show_z=True, show_labels=True, aspect_ratio=1,
        margin=0, scale_up=True, marker_size=None):

    if ax == None:
        ax = plt.gca()
    if aspect_ratio != None:
        ax.set_aspect(aspect_ratio)

    points = np.copy(points)
    if points.shape[1] == 2:
        points = np.pad(points, ((0,0),(0,1)), 'constant')
    if scale_up:
        points *= np.asarray([constants.SENSORS_X, constants.SENSORS_Y, 1])
    points += np.asarray([margin, margin, 0])
    
    for i, p in enumerate(points):
        ax.scatter(p[0], p[1], marker='o', color=col, zorder=10, s=marker_size)
        if show_labels:
            ax.annotate(constants.LABELS[i], (p[0], p[1]), color=col)
        
    if constants.SKELETON_FORMAT == 'clever':
        plot_line(points[1],  points[4],  col=col, ax=ax)
        plot_line(points[4],  points[7],  col=col, ax=ax)
        plot_line(points[7],  points[10], col=col, ax=ax)
        plot_line(points[2],  points[5],  col=col, ax=ax)
        plot_line(points[5],  points[8],  col=col, ax=ax)
        plot_line(points[8],  points[11], col=col, ax=ax)
        plot_line(points[14], points[17], col=col, ax=ax)
        plot_line(points[17], points[19], col=col, ax=ax)
        plot_line(points[19], points[21], col=col, ax=ax)
        plot_line(points[21], points[23], col=col, ax=ax)
        plot_line(points[13], points[16], col=col, ax=ax)
        plot_line(points[16], points[18], col=col, ax=ax)
        plot_line(points[18], points[20], col=col, ax=ax)
        plot_line(points[20], points[22], col=col, ax=ax)
        plot_line(points[12], points[15], col=col, ax=ax)
        plot_line(points[1], points[13], col=col, ax=ax)
        plot_line(points[2], points[14], col=col, ax=ax)
        plot_line(points[1], points[2], col=col, ax=ax)
        plot_line(points[13], points[12], col=col, ax=ax)
        plot_line(points[14], points[12], col=col, ax=ax)
    elif constants.SKELETON_FORMAT == 'hrl-ros':
        plot_line(points[0],  points[1],  col=col, ax=ax)
        plot_line(points[1],  points[2],  col=col, ax=ax)
        plot_line(points[2],  points[4],  col=col, ax=ax)
        plot_line(points[1],  points[3],  col=col, ax=ax)
        plot_line(points[3],  points[5],  col=col, ax=ax)
        plot_line(points[1],  points[6],  col=col, ax=ax)
        plot_line(points[6],  points[8],  col=col, ax=ax)
        plot_line(points[1],  points[7],  col=col, ax=ax)
        plot_line(points[7],  points[9],  col=col, ax=ax)
    elif constants.SKELETON_FORMAT == 'slp':
        plot_line(points[0],  points[1],  col=col, ax=ax)
        plot_line(points[1],  points[2],  col=col, ax=ax)
        plot_line(points[3],  points[4],  col=col, ax=ax)
        plot_line(points[4],  points[5],  col=col, ax=ax)
        plot_line(points[2],  points[3],  col=col, ax=ax)
        plot_line(points[2],  points[8],  col=col, ax=ax)
        plot_line(points[3],  points[9],  col=col, ax=ax)
        plot_line(points[9],  points[12],  col=col, ax=ax)
        plot_line(points[8],  points[12],  col=col, ax=ax)
        plot_line(points[12],  points[13],  col=col, ax=ax)
        plot_line(points[6],  points[7],  col=col, ax=ax)
        plot_line(points[7],  points[8],  col=col, ax=ax)
        plot_line(points[9],  points[10],  col=col, ax=ax)
        plot_line(points[10],  points[11],  col=col, ax=ax)
    elif constants.SKELETON_FORMAT == 'common':
        plot_line(points[0],  points[1],  col=col, ax=ax)
        plot_line(points[1],  points[2],  col=col, ax=ax)
        plot_line(points[3],  points[4],  col=col, ax=ax)
        plot_line(points[4],  points[5],  col=col, ax=ax)
        plot_line(points[2],  points[3],  col=col, ax=ax)
        plot_line(points[2],  points[8],  col=col, ax=ax)
        plot_line(points[3],  points[9],  col=col, ax=ax)
        plot_line(points[9],  points[12],  col=col, ax=ax)
        plot_line(points[8],  points[12],  col=col, ax=ax)
        plot_line(points[6],  points[7],  col=col, ax=ax)
        plot_line(points[7],  points[8],  col=col, ax=ax)
        plot_line(points[9],  points[10],  col=col, ax=ax)
        plot_line(points[10],  points[11],  col=col, ax=ax)
    elif constants.SKELETON_FORMAT == 'arms-omitted':
        plot_line(points[0],  points[1],  col=col, ax=ax)
        plot_line(points[1],  points[2],  col=col, ax=ax)
        plot_line(points[3],  points[4],  col=col, ax=ax)
        plot_line(points[4],  points[5],  col=col, ax=ax)
        plot_line(points[2],  points[3],  col=col, ax=ax)
        plot_line(points[2],  points[6],  col=col, ax=ax)
        plot_line(points[3],  points[7],  col=col, ax=ax)
        plot_line(points[6],  points[8],  col=col, ax=ax)
        plot_line(points[7],  points[8],  col=col, ax=ax)

    return ax

def plot_mattress(image, ax=None, margin=0, border=None, invert_y=True, **kwargs):
    image = np.pad(image, ((margin, margin), (margin, margin)), 'constant')
    image = np.flip(image, axis=0)
    kwargs['square'] = kwargs['square'] if 'square' in kwargs else True
    ax = seaborn.heatmap(image, ax=ax, **kwargs)
    if invert_y:
        ax.invert_yaxis()

    if border != None:
        ax.plot(
            np.array([0, constants.SENSORS_X, constants.SENSORS_X, 0, 0]) + margin,
            np.array([0, 0, constants.SENSORS_Y, constants.SENSORS_Y, 0]) + margin,
            color=border)

    return ax

def plot_results(image, skeletons, ax=None, skeleton_col=['white', 'LightGreen', 'cyan'], skeleton_marker_size=None,
        show_mattress=True, mattress_col='blue', margin=5):

    ax = plot_mattress(image, ax=ax, margin=margin, border=mattress_col) 
    for (i, points) in enumerate(skeletons):
        
        ax = plot_skeleton(points, col=skeleton_col[i % len(skeleton_col)], ax=ax,
                show_z=False, show_labels=False, margin=margin, marker_size=skeleton_marker_size)
    
    return ax
