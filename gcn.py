import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)

    def forward(self, input, adj):
        support = self.weight(input)
        output = torch.bmm(adj, support)
        output = self.layer_norm(output)
        return output


class GCN(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers>0:
            self.layers.append(GraphConvolution(input_size, hidden_size))
            for i in range(num_layers - 1):
                self.layers.append(GraphConvolution(hidden_size, hidden_size))
            self.layers.append(GraphConvolution(hidden_size, num_classes))
        else:
            self.layers.append(GraphConvolution(input_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x, adj)))
        return x
