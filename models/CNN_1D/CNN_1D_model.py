import torch
from torch import nn

class CNN_model(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN_model, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # 计算全连接层的输入大小
        self.flatten_size = self._get_flatten_size(n_timesteps, n_features)
        self.flatten = nn.Flatten()
        # 定义全连接层
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, n_outputs)
        
        # 定义激活函数
        self.relu = nn.ReLU()
    def forward(self, x):
        # 卷积层
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape([-1,x.shape[2],x.shape[1]])
        # 展平
        x = self.flatten(x)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _get_flatten_size(self, n_timesteps, n_features):
        # 计算卷积和池化后的输出大小
        with torch.no_grad():
            x = torch.randn(1, n_features, n_timesteps)
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            # x = self.pool3(self.conv3(x))
            return x.view(1, -1).size(1)

# 创建模型实例
mymodel = CNN_model(7, 1, 7)

# 输入数据
x = torch.ones((1099, 1, 7))  # 注意：输入形状应为 (batch_size, n_features, n_timesteps)
print("输入数据形状:", x.shape)

# 前向传播
y = mymodel(x)
print("输出数据形状:", y.shape)