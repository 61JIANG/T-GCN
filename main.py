import argparse

from utils import *
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from tgcn import *

# parameter
parser = argparse.ArgumentParser(description='S-GCN')
parser.add_argument('--task', default='train', help='train or test')
parser.add_argument('--epoches', default=500, help='Number of epochs to train')
parser.add_argument('--seq_len', default=12, help='time length of input')
parser.add_argument('--pre_len', default=1, help='time length of prediction')
parser.add_argument('--train_len', default=0.6, help='train length of data')
parser.add_argument('--val_len', default=0.2, help='validation length of data')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--layer_num', default=2, help='layers of gru')
parser.add_argument('--out_dim', default=64, help='out dim of gcn')
parser.add_argument('--units', default=8, help='gru out dim')
parser.add_argument('--lr', default=1e-3, help='learning rate')
parser.add_argument('--data', default='los', help='dataset, los or sz')
args = parser.parse_args()

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Graph & data
train, val, test, adj = load_data(args.data, args.train_len, args.val_len)
D = adj.sum(1)
D = torch.diag(torch.pow(D, -0.5))
A = D.mm(adj).mm(D).to(device)

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
mean = np.mean(train)
std = np.std(train)


train_x, train_y = transform_data(train, args.seq_len, args.pre_len, device)
val_x, val_y = transform_data(val, args.seq_len, args.pre_len, device)
test_x, test_y = transform_data(test, args.seq_len, args.pre_len, device)

# dataloader
train_data = torch.utils.data.TensorDataset(train_x, train_y)
train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(val_x, val_y)
val_iter = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False)
test_data = torch.utils.data.TensorDataset(test_x, test_y)
test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False)

# Loss & Model  &Optimizer
loss_criterion = nn.L1Loss()
model = tgcn(A, train_x.shape[1], args.out_dim, args.seq_len, args.pre_len, args.layer_num, args.units).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train():
    if args.task=='train':
        for epoch in range(args.epoches):
            model.train()
            for x, y in train_iter:
                y_pred = model(x)
                loss = loss_criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss.item())
            rmse = evaluate(val_iter,model)
            print('epoch:{}, rmse={}'.format(epoch, rmse))


def evaluate(test_iter, model):
    model.eval()
    rmse = 0.0
    for x, y in test_iter:
        y_pred = model(x)
        y_pred, y = y_pred * std + mean, y * std + mean
        rmse += torch.sum(torch.pow(y_pred - y, 2)) / (x.shape[0]*x.shape[1]*x.shape[2])
    return torch.sqrt(rmse)


if __name__ == '__main__':
    train()
