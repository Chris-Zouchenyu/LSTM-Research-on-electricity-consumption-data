#导包
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential,Conv1d,MaxPool1d,Linear,Flatten,ReLU,Dropout1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#模型搭建
class CNN(nn.Module):
    '''
    This is a new model
    window_size:
    fea_num:

    '''
    def __init__(self,n_timesteps, n_features, n_outputs):
        super().__init__()
        self.model1 = Sequential(
            Conv1d(in_channels = n_features, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            ReLU(),
            Conv1d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
            ReLU(),
            MaxPool1d(kernel_size = 2),
            Conv1d(in_channels = 64, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
            ReLU(),
            MaxPool1d(kernel_size = 2),
            Flatten(),
            Linear(in_features=128,out_features = 100),
            ReLU(),
            Linear(in_features=100,out_features=n_outputs)
        )
        
    def forward(self,x):
        x = self.model1(x)
        return x

# n_timesteps, n_features, n_outputs = 7,8,7
# mymodel = CNN_LSTM(n_timesteps, n_features, n_outputs)
# test1 = torch.rand((99,8,7)) # 注意：输入形状应为 (batch_size, n_features, n_timesteps)
# print(test1.shape)
# print((mymodel(test1.float())).shape)# (99,8,7) -> (99,7)