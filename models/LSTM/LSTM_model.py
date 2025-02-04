import torch
from torch.nn import LSTM,ReLU,Linear

class LSTM_model(torch.nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lstm1 = LSTM(input_size = n_features, hidden_size = 200, num_layers = 1, batch_first = True)
        self.relu1 = ReLU()
        self.linear1 = Linear(in_features = 200, out_features = 1)
    def forward(self,x):
        x,(hn,cn) = self.lstm1(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x = x.reshape([x.shape[0],x.shape[1]*x.shape[2]])
        return x

# 测试一下
# x = torch.rand((1099,7,1))
# model = LSTM_model(1,7)
# y = model(x)
# print(x,y,x.shape,y.shape)

    