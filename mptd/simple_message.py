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

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import spmm

class SimpleMessage(MessagePassing):
    """
    Implements a simple message passing layer for graph neural networks.

    Parameters
    ----------
    aggr : str, optional
        Aggregation method to use ('add', 'mean', 'max', etc.), by default 'add'.

    Returns
    -------
    None
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """
        Forward pass of the message passing layer.

        Parameters
        ----------
        x : Tensor
            Node feature tensor.
        edge_index : Adj
            Graph edge indices.
        edge_weight : OptTensor, optional
            Edge weights, by default None.

        Returns
        -------
        Tensor
            Result of the message passing operation.
        """
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        """
        Message function used in message passing.

        Parameters
        ----------
        x_j : Tensor
            Node feature tensor for neighboring nodes.
        edge_weight : OptTensor
            Edge weights, by default None.

        Returns
        -------
        Tensor
            The modified node feature tensor for neighboring nodes.
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Message and aggregation function used in message passing.

        Parameters
        ----------
        adj_t : SparseTensor
            Transposed adjacency matrix.
        x : Tensor
            Node feature tensor.

        Returns
        -------
        Tensor
            The aggregated node feature tensor after message passing.
        """
        return spmm(adj_t, x, reduce=self.aggr)
