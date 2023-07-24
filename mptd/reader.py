# DataPreprocessing/utilities.py
# Copyright (c) 2022-present Samuele Colombo
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
import torch

import astropy.table as astropy_table

from astropy import units as u
from astropy.io import fits
from astropy.table import Table

newunits = [u.def_unit("PIXELS", u.pixel),
            u.def_unit("CHAN", u.chan),
            u.def_unit("CHANNEL", u.chan),
            u.def_unit("0.05 arcsec", 0.05*u.arcsec)
           ]

def read_events(genuine, simulated, keys):
    """
    Reads events from a genuine file and a simulated file, and removes the duplicate events.
    The function also adds a new column indicating whether the event is simulated.

    Parameters
    ----------
    genuine : str, file-like object, list, pathlib.Path object
        File containing only observed events.
    simulated : str, file-like object, list, pathlib.Path object
        File containing observed events plus simulated ones.
    keys : str or list of str
        Column labels for the attributes of interest

    Returns
    -------
    astropy.Table
        A table containing the attributes selected through the `keys` parameter
        and the "ISSIMULATED" value for each event, indicating whether the event is simulated or not.
    """
    # Read the genuine and simulated events
    with u.add_enabled_units(newunits), \
         fits.open(genuine, memmap=True) as gen_file, \
         fits.open(simulated, memmap=True) as sim_file:
        F_dat = Table(sim_file[1].data)
        if 'ISSIMULATED' in F_dat.colnames:
            keys = list(keys) + ['ISSIMULATED']
            return F_dat[keys]
        I_dat = Table(gen_file[1].data)

    # Join the genuine and simulated events and remove the duplicate events
    dat = astropy_table.vstack([I_dat, F_dat], join_type='exact')
    dat = astropy_table.unique(dat, keys=keys, keep='none')

    # num_simulated = len(dat) - len(I_dat)
    num_simulated = len(dat)

    # Add a new column indicating whether the event is simulated
    dat['ISSIMULATED']   = astropy_table.Column([True] * num_simulated, dtype=bool)
    I_dat['ISSIMULATED'] = astropy_table.Column([False] * len(I_dat), dtype=bool)
    dat = astropy_table.vstack([dat, I_dat], join_type="exact")

    # Select only the columns specified in the `keys` parameter and the "ISSIMULATED" column
    keys = list(keys) + ['ISSIMULATED']
    return dat[keys]

def filter_from_key(data, key, low, high):
    """
    Filter data based on a specified key within a range.

    Parameters
    ----------
    data : astropy.Table
        The data table to filter.
    key : str
        The key to filter on.
    low : float
        The lower bound of the range.
    high : float
        The upper bound of the range.

    Returns
    -------
    np.ndarray
        Boolean array indicating whether each row satisfies the filter.
    """
    return np.logical_and(data[key] >= low, data[key] <= high)

def get_raw_data(filename, keys, filters:dict):
    """
    Extract raw data from a file based on specified keys and filters.

    Parameters
    ----------
    filename : str, file-like object, list, pathlib.Path object
        The file containing the data.
    keys : List[str]
        Column labels for the attributes of interest.
    filters : dict
        Dictionary containing filtering conditions for specified keys.

    Returns
    -------
    astropy.Table
        A table containing the raw data with selected keys and "ISSIMULATED" column.
    """
    keys_plus = keys + [key for key in filters.keys() if key not in keys]
    if 'EVLI' in filename:
        companion = filename.replace('EVLI', 'EVLF')
        raw_data = read_events(filename, companion, keys_plus)
    elif 'EVLF' in filename:
        companion = filename.replace('EVLF', 'EVLI')
        raw_data = read_events(companion, filename, keys_plus)
    else:
        raise Exception("filename does not contain the 'EVLI' or 'EVLF' indicator in the file name.")
    for key, values in filters.items():
        raw_data = raw_data[filter_from_key(raw_data, key, *values)]
    return raw_data[keys + ["ISSIMULATED"]]
