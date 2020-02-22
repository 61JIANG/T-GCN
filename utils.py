import torch
import numpy as np
import pandas as pd


def load_data(data, train_len, val_len):
    """
    load speed data adnd adj from file_path
    """
    if data == 'los':
        data_path = './data/los_speed.csv'
        adj_path = './data/los_adj.csv'
    if data == 'sz':
        data_path = './data/sz_speed.csv'
        adj_path = './data/sz_adj.csv'
    data = pd.read_csv(data_path).values.astype(float)
    data_len = len(data)
    train_len = int(train_len * data_len)
    val_len = int(val_len * data_len)
    train = data[:train_len]
    val = data[train_len:train_len + val_len]
    test = data[train_len+val_len:]

    adj = pd.read_csv(adj_path, header=None).values.astype(float)
    adj = torch.Tensor(adj)

    return train, val, test, adj


def transform_data(data, num_timestep_input, num_timestep_pred, device):

    n_slot = len(data) - (num_timestep_input + num_timestep_pred) + 1
    indices = [(i, i + num_timestep_input + num_timestep_pred) for i in range(n_slot)]
    x = np.zeros([n_slot, 1, num_timestep_input, data.shape[1]])
    y = np.zeros([n_slot, num_timestep_pred, data.shape[1]])
    for i, j in indices:
        x[i, :, :, :] = data[i:i+num_timestep_input].reshape(1,num_timestep_input,data.shape[1])
        y[i, :, :] = data[i+num_timestep_input:j].reshape(num_timestep_pred, data.shape[1])
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device).permute(0, 2, 1)
    return x, y
