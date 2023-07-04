import torch

import torch_geometric.transforms as ttr

from mptd.reader import get_data
from mptd.simple_message import SimpleMessage
from mptd.plotter import plot_data, plot_clusters
from mptd.clusterer import get_clusters

def main(filename, keys, k, layers, quantile):
    data = get_data(filename=filename, keys=keys, filters={"FLAG": (0,4)}).cuda()

    transform = ttr.KNNGraph(k=k, force_undirected=True)

    net_data =transform(data)

    distances = torch.norm(net_data.x[net_data.edge_index[0]] - net_data.x[net_data.edge_index[1]], dim=1)

    model = SimpleMessage()

    issimulated=data.y.bool()

    masked_data, masked_sizes, labels, mask = \
        get_clusters(net_data, distances, model, layers, quantile)

    data = data.cpu()
    issimulated = issimulated.cpu()

    plot_data(data.x, (0.01, 0.5), issimulated, keys)
    plot_data(masked_data, masked_sizes, issimulated[mask], keys)
    plot_clusters(masked_data[:, 1:4], masked_sizes, labels, keys[1:4])
    plot_data(data.x, (0.0, 0.5), issimulated, keys)

if __name__ == "__main__":
    import os.path as osp

    # filename = osp.join("test.onD", "Icaro", "raw", "0690751601", "pps", "P0690751601M2S002MIEVLF0000.FTZ") # easy example
    filename = osp.join("test.onD", "Icaro", "raw", "0694730101", "pps", "P0694730101PNS003PIEVLF0000.FTZ") # hard example
    # filename = osp.join("test.onD", "Icaro", "raw", "0744440301", "pps", "P0744440301M1S001MIEVLF0000.FTZ") # medium example
    keys = ["PI", "TIME", "X", "Y"]
    k = 8
    layers = 10
    quantile = 0.99
    main(filename, keys, k, layers, quantile)
