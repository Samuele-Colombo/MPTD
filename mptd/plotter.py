# Copyright (c) 2023-present Samuele Colombo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transient_detection.DataPreprocessing.utilities import read_events

def plot_fits_data(filename, outfile=None, sizes=None):
    """
    Plots data from a FITS file containing observed events and simulated ones.
    The events are displayed in a 3D scatter plot with the X, Time, and Y coordinates.
    Background events are shown in blue, and transient events are shown in red.
    The size of the points representing events are determined by the respective 'PI' attributes in the file.

    Parameters
    ----------
    filename : str
        The path to the FITS file containing the data.
    outfile : str, optional
        The path to save the plot as an image file. If not provided, the plot will be displayed interactively.
    sizes : numpy.ndarray or None, optional
        An array of sizes for the points representing the events. If None, the size will be determined based on the 'PI' attribute.

    Returns
    -------
    None
    """
    # Read the FITS file and extract the data
    print("Opening file...")
    with fits.open(filename) as hdul:
        data = hdul[1].data
    
    if 'ISEVENT' not in data.columns.names:
        if 'EVLI' in filename:
            companion = filename.replace('EVLI', 'EVLF')
            data = read_events(filename, companion, ['X', 'Y', 'TIME', 'PI'])
        elif 'EVLF' in filename:
            companion = filename.replace('EVLF', 'EVLI')
            data = read_events(companion, filename, ['X', 'Y', 'TIME', 'PI'])
        else:
            raise Exception("filename does not contain the 'ISEVENT' colname and has not the 'EVLI' or 'EVLF' indicator in the file name. Check file integrity")
        data.rename_column('ISSIMULATED', 'ISEVENT')

    data['PI'] = np.log2(((data['PI'] - data['PI'].min())/(data['PI'].max() - data['PI'].min())) + 2) * 10

    # Extract the individual columns
    is_event = data['ISEVENT']
    
    # Separate the data points based on the label (background or event)
    background_points = data[is_event == 0]
    event_points = data[is_event == 1]
    
    # Set the size of the points based on PI
    if sizes is None:
        background_sizes = background_points['PI'] if sizes is None else sizes
        event_sizes = event_points['PI'] if sizes is None else sizes
    elif len(sizes) == 1:
        background_sizes = event_sizes = sizes
    else:
        background_sizes = sizes[0]
        event_sizes = sizes[1]

    print("Plotting...")
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the background points
    ax.scatter(background_points['X'], background_points['TIME'], background_points['Y'],
               c='b', label='Background', alpha=0.6, s=background_sizes)
    
    # Plot the event points
    ax.scatter(event_points['X'], event_points['TIME'], event_points['Y'],
               c='r', label='Event', alpha=0.6, s=event_sizes)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Time')
    ax.set_zlabel('Y')

    import os.path as osp
    
    ax.set_title(f"Transient from '{osp.basename(filename)}'\n {len(background_points)} bkg events, {len(event_points)} transient events.")

    # Add a legend
    ax.legend()
    
    # Show the plot
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()

def plot_data(data, color, issimulated, keys, title=None, outfile=None):
    """
    Plots data in a 3D scatter plot with X, Time, and Y coordinates.
    The points are colored based on whether they are simulated or not (issimulated).
    The size of the points can be adjusted based on the provided 'color' parameter.

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot.
    color : numpy.ndarray or tuple
        The color or size information for the points. If a tuple is provided, it represents the size of background and simulated points.
    issimulated : numpy.ndarray
        Boolean array indicating whether each data point is simulated (True) or not (False).
    keys : List[str]
        Column labels for the attributes of interest.
    title : str, optional
        The title of the plot.
    outfile : str, optional
        The path to save the plot as an image file. If not provided, the plot will be displayed interactively.

    Returns
    -------
    None
    """
    pd_data = pd.DataFrame(data , columns=keys)

    pd_data["ISSIMULATED"] = issimulated
    issimulated = pd_data["ISSIMULATED"]

    plot_bkg = True
    if len(color) > 2:
        event_sizes = (color - color.min())/(color.max() - color.min())
        bkg_sizes = event_sizes[~issimulated]
        sim_sizes = event_sizes[issimulated]
    else:
        bkg_sizes, sim_sizes = color
        if bkg_sizes == 0:
            plot_bkg = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the background points
    bkg_data = pd_data[~issimulated]
    if plot_bkg:
        ax.scatter(bkg_data['X'], bkg_data['TIME'], bkg_data['Y'],
                   c="blue", alpha=0.6, s=bkg_sizes, label="background")

    sim_data = pd_data[issimulated]
    ax.scatter(sim_data['X'], sim_data['TIME'], sim_data['Y'],
                c="red", alpha=0.6, s=sim_sizes, label="simulated")
    
    # Set axis labels
    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    ax.set_zlabel(keys[2])

    # Add a legend
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    plt.show()

def plot_clusters(data, sizes, labels, keys, **fig_kwargs):
    """
    Plots data points in a 3D scatter plot and colors the points based on cluster labels.

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot.
    sizes : numpy.ndarray
        The size of points for each data point.
    labels : numpy.ndarray
        The cluster labels for each data point.
    keys : List[str]
        Column labels for the attributes of interest.
    fig_kwargs : dict
        Additional keyword arguments for creating the figure.

    Returns
    -------
    None
    """
    # Create a 3D scatterplot
    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each data point with a color corresponding to its cluster label
    for label in np.unique(labels):
        if label == -1: continue
        label_mask = labels==label
        labeled_data = data[label_mask]
        if len(labeled_data) > 0:
            ax.scatter(labeled_data[:, 0], labeled_data[:, 1], labeled_data[:, 2], label=label, sizes=sizes)

    # Set labels for each axis
    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    ax.set_zlabel(keys[2])

    # Show the plot
    plt.show()
