import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class gcn(nn.Module):
    """Spatial Dependence Modeling"""
    def __init__(self, A, in_features, out_features, bias=False):
        super(gcn, self).__init__()
        self.A = A
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.w1 = nn.Parameter(torch.FloatTensor(self.out_features, out_features//4))
        if bias:
            pass
        else:
            pass
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w0.shape[1])
        self.w0.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.w1.shape[1])
        self.w1.data.uniform_(-stdv1, stdv)

    def forward(self, x):
        x = x.permute(3, 0, 2, 1)
        x = torch.einsum('mn, nbti->bmti', self.A, x)
        x = F.relu(torch.einsum('bmti,io->bmto', x, self.w0))
        x = torch.einsum('mn, nbti->bmti', self.A, x.permute(1, 0, 2, 3))
        x = F.sigmoid(torch.einsum('bmti,io->bmto', x, self.w1))
        x = x.permute(0, 3, 2, 1)
        return x


class gru(nn.Module):
    """
    Temporal Dependence Modeling
    """

    def __init__(self, in_features, num_timestep_input, layer_num, out_features):
        super(gru, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_timestep_input = num_timestep_input
        self.layer_num = layer_num
        self.gru = nn.GRU(in_features//4, out_features, layer_num)

    def forward(self, x):
        x = x.permute(3, 2, 0, 1)
        temp = []
        for i in range(x.shape[0]):
            output, hn = self.gru(x[i])
            temp.append(output[self.num_timestep_input - 1])
        temp = torch.cat(temp, dim=0).reshape(x.shape[0], x.shape[2], self.out_features)
        temp = F.relu(temp.permute(1, 0, 2))
        return temp


class tgcn(nn.Module):
    """
    tgcn(A, train_x.shape[1], args.out_dim, args.seq_len, args.pre_len, args.layer_num).to(device)
    Temporal Graph Convolutional Network
    input data:(batch_size, num_features, seq_len, num_nodes)
    """
    #tgcn(A, train_x.shape[1], args.out_dim, args.seq_len, args.pre_len, args.layer_num).to(device)
    def __init__(self, A, in_features, out_features, num_timestep_input, num_timestep_pred, layer_num, units):
        super(tgcn, self).__init__()
        self.A = A
        self.in_features = in_features
        self.num_timestep_input = num_timestep_input
        self.num_tiemstep_pred = num_timestep_pred
        self.out_features = out_features
        self.layer_num = layer_num
        self.units = units
        self.gcn = gcn(self.A, self.in_features, self.out_features)
        self.gru = gru(self.out_features, self.num_timestep_input, self.layer_num, self.units)
        self.fully = nn.Linear(self.units, self.num_tiemstep_pred)

    def forward(self, x):
        x = self.gcn(x)
        x = self.gru(x)
        fx = self.fully(x.reshape((x.shape[0]*x.shape[1], x.shape[2])))
        x = fx.reshape(x.shape[0], x.shape[1], self.num_tiemstep_pred)
        return x