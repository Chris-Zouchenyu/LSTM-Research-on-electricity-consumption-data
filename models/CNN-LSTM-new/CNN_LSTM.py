import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
import torchkeras
from torch.utils.data import TensorDataset,DataLoader
from CNN_LSTM_model import *

def split_dataset(data):
    '''
    该函数实现以周为单位切分训练数据和测试数据
    '''
    # data为按天的耗电量统计数据，shape为(1442, 8)
    # 测试集取最后一年的46周（322天）数据，剩下的159周（1113天）数据为训练集，以下的切片实现此功能。
    train, test = data[1:-328], data[-328:-6]
    train = np.array(np.split(train, len(train)/7)) # 将数据划分为按周为单位的数据
    test = np.array(np.split(test, len(test)/7))
    return train, test

def evaluate_forecasts(actual, predicted):
    '''
    该函数实现根据预期值评估一个或多个周预测损失
    思路：统计所有单日预测的 RMSE
    '''
    scores = list()
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
    
    s = 0 # 计算总的 RMSE
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))
    return score, scores

def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s\n' % (name, score, s_scores))
    
def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    '''
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2])) # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []
    
    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out
        
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, 0]
            train_seq = train_seq.reshape((len(train_seq), 1))
            X.append(train_seq)
            y.append(data[in_end:out_end, 0])
        in_start += 1
        
    return np.array(X), np.array(y)

def cnn_lstm_model(train, sw_width, in_start=0, verbose_set=0, epochs_num=20, batch_size_set=4):
    '''
    该函数定义 Encoder-Decoder LSTM 模型
    '''
    train_x, train_y = sliding_window(train, sw_width, in_start=0)
    train_x = torch.tensor(train_x, dtype = torch.float32)
    train_y = torch.tensor(train_y, dtype = torch.float32)
    
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    train_x = train_x.transpose(-1,-2)
    # 数据加载
    ds_train = TensorDataset(train_x, train_y)
    dl_train = DataLoader(ds_train, batch_size=4, num_workers=0)

    # torchkeras训练
    model = torchkeras.Model(CNN_LSTM_model(n_features,n_outputs,n_timesteps))
    # pytorch训练
    # model = LSTM_model(n_features,n_outputs)

    # 超参数
    lr = 0.001
    EPOCH = 30
    batch_size = 4
    # 损失函数
    loss_fn = torch.nn.MSELoss()
    loss_seq = []
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # torchkeras训练
    model.compile(loss_func=loss_fn, optimizer=optimizer)
    # model.summary()
    model.fit(epochs=EPOCH,dl_train=dl_train)
    
    
    return model

def forecast(model, pred_seq, sw_width):
    '''
    该函数实现对输入数据的预测
    '''
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
    input_x = data[-sw_width:, 0] # 获取输入数据的最后一周的数据
    input_x = input_x.reshape((1, len(input_x), 1)) # 重塑形状[1, sw_width, 1]
    input_x = torch.tensor(input_x,dtype=torch.float32)
    input_x = input_x.transpose(-1,-2)

    yhat = model(input_x) # 预测下周数据
    yhat = yhat[0] # 获取预测向量
    return yhat

def evaluate_model(model, train, test, sd_width):
    '''
    该函数实现模型评估
    '''
    history_fore = [x for x in train]
    predictions = list() # 用于保存每周的前向验证结果；
    with torch.no_grad():
        for i in range(len(test)):
            yhat_sequence = forecast(model, history_fore, sd_width) # 预测下周的数据
            predictions.append(yhat_sequence) # 保存预测结果
            history_fore.append(test[i, :]) # 得到真实的观察结果并添加到历史中以预测下周
        
        predictions = np.array(predictions) # 评估一周中每天的预测结果
        score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

def model_plot(score, scores, days, name):
    '''
    该函数实现绘制RMSE曲线图
    '''
    plt.figure(figsize=(8,6), dpi=150)
    plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('CNN-LSTM 模型预测结果',  size=18)
    plt.legend()
    plt.show()
    
def main_run(dataset, sw_width, days, name, in_start, verbose, epochs, batch_size):
    '''
    主函数：数据处理、模型训练流程
    '''
    # 划分训练集和测试集
    train, test = split_dataset(dataset.values)
    # 训练模型
    model = cnn_lstm_model(train, sw_width, in_start, verbose_set=0, epochs_num=20, batch_size_set=4)
    # 计算RMSE
    score, scores = evaluate_model(model, train, test, sw_width)
    # 打印分数
    summarize_scores(name, score, scores)
    # 绘图
    model_plot(score, scores, days, name)
    
    print('------头发不够，帽子来凑-----')
    
    
if __name__ == '__main__':
    
    dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
                   infer_datetime_format=True, engine='c',
                   parse_dates=['datetime'], index_col=['datetime'])
    # 数据归一化处理，使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
    for i in range(dataset.shape[1]):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
    
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    name = 'CNN-LSTM'
    
    sliding_window_width= 7
    input_sequence_start=0
    
    epochs_num=20
    batch_size_set=16
    verbose_set=0
    
    
    main_run(dataset, sliding_window_width, days, name, input_sequence_start,
             verbose_set, epochs_num, batch_size_set)
