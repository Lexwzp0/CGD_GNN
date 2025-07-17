import torch.nn as nn
from torch_geometric.nn import  global_mean_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch.nn import Linear, Dropout


def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred == y_true) / len(y_true)


class GAT(torch.nn.Module):
    def __init__(self, in_feats,hidden_channels, out_feats, num_layers, heads=8):
        super(GAT, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.heads = heads

        # Initialize graph convolutional layers
        self.pre_Linear = nn.Linear(in_feats, hidden_channels)
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels,heads=heads)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels,heads=heads)
        self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels,heads=heads)
        self.classifier = nn.Linear(hidden_channels*heads,out_feats )


    def forward(self, x, edge_index, batch):
        x = self.pre_Linear(x)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x= self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x,training=self.training)
        x= self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x= self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
