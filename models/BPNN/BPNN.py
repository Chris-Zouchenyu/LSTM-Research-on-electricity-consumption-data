import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchkeras
from BPNN_model import *

torch.manual_seed(1234)
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

from torch.utils.data import TensorDataset, DataLoader
import math
import sklearn.metrics as skm
'''
这些模型会在家庭用电量数据集上进行演示。
通过之前的模型可知，如果一个模型比一个七天预测朴素模型的RMSE小，就可以认为该模型是可用的。
'''
# 使用重采后的数据，将原来每分钟采集的数据转化为每小时采集的数据

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

# print(train.shape)
def sliding_window(train, sw_width=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    '''
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2])) # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []
    
    for _ in range(len(data)):
        in_end = in_start + sw_width# 0+7 1+7 2+7
        out_end = in_end + sw_width# 7+7 8+7 9+7
        
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, 0]
            train_seq = train_seq.reshape((len(train_seq), 1))
            X.append(train_seq)# 0:7 1:8 2:9
            y.append(data[in_end:out_end,0])# 7:14 8:15 9:16
        in_start += 1
    return np.array(X), np.array(y)
# X,y = sliding_window(train,sw_width=7,in_start=0)# 这里的X是每一天的总电能数据
# X = X.reshape((X.shape[0],1,-1))
# print(X.shape,y.shape)# X:(1099,1,7), y:(1099,7)

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
    s_scores = ', '.join(['%.4f' % s for s in scores])
    print('%s: [%.4f] %s\n' % (name, score, s_scores))

def cnn_model(train, sw_width, epochs_num, in_start=0, verbose_set=0, batch_size=4):
    '''
    该函数调用1D CNN模型
    '''
    mode = 'train'
    train_x, train_y = sliding_window(train,sw_width,in_start)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_x = torch.tensor(train_x, dtype = torch.float32)
    train_y = torch.tensor(train_y, dtype = torch.float32)
    train_x = train_x.reshape((train_x.shape[0],1,-1))
    
    # torchkeras训练
    model = torchkeras.Model(CNN_model(n_timesteps, n_features, n_outputs))
    ds_train = TensorDataset(train_x, train_y)
    dl_train = DataLoader(ds_train, batch_size=epochs_num, num_workers=0)
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # 损失函数
    loss_fn = nn.MSELoss()
    model.compile(loss_func=loss_fn, optimizer=optimizer)
    model.fit(epochs=epochs_num,dl_train=dl_train)

    # pytorch训练
    # model = CNN_model(n_timesteps, n_features, n_outputs)
    # # 训练准备
    # # 优化器
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # # 损失函数
    # loss_fn = nn.MSELoss()
    # # 开始训练
    # for epoch in range(epochs_num):
    #     loss_train = 0
    #     for num in range(37):
    #         i = 0
    #         datax = train_x[i:batch_size+i,:,:]
    #         datay = train_y[i:batch_size,:]
    #         model.train()
    #         y_hat = model(datax)
    #         loss = loss_fn(y_hat,datay)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_train += loss.item()
    #         i+=batch_size
    #     print("[TRAIN] =========epoch : {},  loss  {:.4f}======== ".format(epoch+1,loss_train))
    return model

def forecast(model, pred_seq, sw_width):
    '''
    该函数实现对输入数据的预测
    '''
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
    input_x = data[-sw_width:, 0] # 获取输入数据的最后一周的数据
    input_x = input_x.reshape((1, 1, len(input_x))) # 重塑形状[1, 1, sw_width]
    input_x = torch.tensor(input_x,dtype=torch.float32)
    # input_x = F.normalize(input_x)


    yhat = model(input_x) # 预测下周数据 verbose???
    yhat = yhat[0] # 获取预测向量
    return yhat

def evaluate_model(model, train, test, sd_width):
    '''
    该函数实现模型评估
    '''
    model.eval()
    history_fore = [x for x in train]
    predictions = list() # 用于保存每周的前向验证结果；
    with torch.no_grad():
        for i in range(len(test)):
            yhat_sequence = forecast(model, history_fore, sd_width) # 预测下周的数据
            predictions.append(yhat_sequence) # 保存预测结果
            history_fore.append(test[i, :]) # 得到真实的观察结果并添加到历史中以预测下周
        predictions = np.array(predictions) # 评估一周中每天的预测结果
        # test_normalized = F.normalize(torch.tensor(test[:, :, 0], dtype=torch.float32))# 将测试集的数据归一化
        score, scores = evaluate_forecasts(test[:, :, 0], predictions)
        print("Expected: {}\nPredictions: {}".format(test[:, :, 0], predictions))
        x = test[:, :, 0]
    return score, scores#, x, predictions



# # 训练小测试
# dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
#                    infer_datetime_format=True, engine='c',
#                    parse_dates=['datetime'], index_col=['datetime'])
# train,test = split_dataset(dataset)
# model = cnn_model(train = train, sw_width = 7,epochs_num=200)
# score, scores, x, predictions = evaluate_model(model, train, test, sd_width = 7)
# # print(test.shape, predictions.shape)(46, 7, 8) (46, 7)
# test = test[:, :, 0]
# test = test.reshape([-1])
# predictions = predictions.reshape([-1])
# # print(test.shape, predictions.shape (322,) (322,)
# plt.figure(figsize=(8,6), dpi=150)
# plt.plot([x for x in range(322)], test, label='test')
# plt.plot([x for x in range(322)], predictions, label='predictions')
# plt.legend()
# plt.show()

def model_plot(score, scores, days, name):
    '''
    该函数实现绘制RMSE曲线图
    '''
    plt.figure(figsize=(8,6), dpi=150)
    plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('BPNN 模型预测结果',  size=18)
    plt.legend()
    plt.show()

def main_run(dataset, sw_width, days, name, in_start, epochs_num, verbose, batch_size):
    '''
    主函数：数据处理、模型训练流程
    '''
    # 划分训练集和测试集
    train, test = split_dataset(dataset.values)
    # 训练模型
    model = cnn_model(train, sw_width, epochs_num, in_start, verbose_set=0)
    # 计算RMSE
    score, scores = evaluate_model(model, train, test, sw_width)
    # 打印分数
    summarize_scores(name, score, scores)
    # 绘图
    model_plot(score, scores, days, name)

if __name__ == '__main__':
    
    dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
                   infer_datetime_format=True, engine='c',
                   parse_dates=['datetime'], index_col=['datetime'])
    
    # 整个数据集按列归一处理
    for i in range(dataset.shape[1]):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    name = 'cnn'
    
    sliding_window_width=7
    input_sequence_start=0
    
    epochs_num=20
    batch_size_set=4
    verbose_set=0
    
    main_run(dataset, sliding_window_width, days, name, input_sequence_start,
             epochs_num, verbose_set, batch_size_set)
