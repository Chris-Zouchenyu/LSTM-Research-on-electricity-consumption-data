import torch
from torch import nn

class CNN_model(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN_model, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 将 (1, 1, 7) 展平为 (1, 7)
            nn.Linear(in_features=7, out_features=32),  # 更改这里的输入特征数为 7
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_outputs),
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=n_outputs),
            # nn.ReLU(),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# # 创建模型实例
# mymodel = CNN_model(7, 1, 7)

# # 输入数据
# x = torch.ones((1099, 1, 7))  # 注意：输入形状应为 (batch_size, n_features, n_timesteps)
# print("输入数据形状:", x.shape)

# # 前向传播
# y = mymodel(x)
# print("输出数据形状:", y.shape)