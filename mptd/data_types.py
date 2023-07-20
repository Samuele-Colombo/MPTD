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

import torch
from torch_geometric.data import Data

class SimTransientData(Data):
    """
    Subclass of `torch_geometric.data.Data` that adds a `pos` property to store and retrieve the position data. The position data is stored in the last three values of the `x` attribute.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    pos : np.ndarray
        3-D position data.
    """
    def __init__(self, x = None, edge_index = None, edge_attr = None, y = None, pos = None, **kwargs):
        assert pos is None, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
                             " values of the `x` attribute. Please append the position coordinates to your 'x' parameter")
        # assert x.shape[1] >= 3, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
        #                          " values of the `x` attribute. Therefore the 'x' parameter must contain at least three elements.")
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
    
    @property
    def pos(self):
        """
        Getter for the position data.
        
        Returns
        -------
        np.ndarray
            3-D position data.
        """
        return self.x[:, -3:]

    @pos.setter
    def pos(self, replace):
        """
        Setter for the position data.
        
        Parameters
        ----------
        replace : np.ndarray
            3-D position data to be set.
        
        Raises
        ------
        AssertionError
            If the shape of `replace` is not (num_points, 3).
        """
        assert replace.shape == self.pos.shape
        self.x[:, -3:] = replace
    
    def append(self, other):
        new_edge_index = torch.hstack([self.edge_index, other.edge_index]) \
                         if other.edge_index is not None and self.edge_index is not None else None
        new_edge_attr = torch.hstack([self.edge_attr, other.edge_attr]) \
                        if other.edge_attr is not None and self.edge_attr is not None else None
        
        return SimTransientData(x=torch.vstack([self.x, other.x]), 
                                y=torch.hstack([self.y, other.y]), 
                                edge_index=new_edge_index,
                                edge_attr=new_edge_attr
                               )