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

from sklearn.cluster import DBSCAN

import torch

def get_clusters(net_data, distances, model, layers, quantile):
    elaborated_data = torch.ones_like(net_data.x[:, 0].unsqueeze(-1))
    # issimulated=net_data.y.bool()

    for i in range(layers):
        elaborated_data += model.forward(elaborated_data, net_data.edge_index)#, edge_weight=gaussian_kernel(distances, distances.median()*1000))
        # Calculate the threshold value (10th percentile)
        sizes = elaborated_data[:,0]
        threshold = torch.quantile(sizes,  quantile).item()
        mask = sizes >= threshold
        # if (i+1)%1 == 0:
        #     plotdata(data.x[mask].cpu(), sizes[mask].cpu(), issimulated=issimulated[mask].cpu(), keys=keys, title=f"iteration {i+1}")
        elaborated_data /= elaborated_data.max()

    dbscan = DBSCAN(eps=distances.median().item()/2, min_samples=5)  # Adjust parameters according to your data

    masked_data = net_data.x[mask].cpu()
    masked_sizes = sizes[mask].cpu()

    labels = dbscan.fit_predict(masked_data)

    return masked_data, masked_sizes, labels, mask.cpu()