import torch
from torch.nn import LSTM,ReLU,Linear,Sequential,Conv1d,Flatten

class CNN_LSTM_model(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_windows):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_windows = n_windows
        self.model1 = Sequential(
            Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            ReLU(),
            Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            ReLU(),
            Flatten(),
        )
        self.linear1 = Linear(in_features=192, out_features=100)
        self.linear2 = Linear(in_features=100, out_features=1)
    def forward(self,x):
        x = self.model1(x)
        xshape = x.shape[0]
        x = x.repeat(self.n_windows,1)
        x = x.reshape([xshape,self.n_windows,-1])
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# # 测试一下
# x = torch.rand((1099,7,1))
# x = x.transpose(-1,-2)
# model = CNN_LSTM_model(1,7,7)
# y = model(x) (1099,1,7) -> (1099,7,1)
# print(x.shape,y.shape)

    