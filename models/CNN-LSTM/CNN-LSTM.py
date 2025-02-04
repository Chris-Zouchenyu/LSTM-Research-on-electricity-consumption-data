import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.metrics as skm
import torch
import torchkeras
import torch.nn.functional as F
from CNN_LSTM_model import *
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(1234)
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

'''
这些模型会在家庭用电量数据集上进行演示。
通过之前的模型可知，如果一个模型比一个总RMSE约为465千瓦的七天预测朴素模型的RMSE小，就可以认为该模型是可用的。
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

def sliding_window(train, sw_width=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据，将训练数据划分为两组，用除总电能以外的数据，去预测这总电能这个数据
    '''
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2])) # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []
    
    for _ in range(len(data)):
        in_end = in_start + sw_width# 0+7 1+7 2+7
        out_end = in_end + sw_width# 7+7 8+7 9+7
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_x_seq = data[in_start:in_end, 1:]
            train_y_seq = data[in_start:in_end, 0]
            X.append(train_x_seq)# 0:7 1:8 2:9
            y.append(train_y_seq)# 0:7 1:8 2:9
        in_start += 1
    return np.array(X),np.array(y)

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

def cnn_model(train, EPOCH, lr, batch_size, window_size, fea_num=7):
    '''
    该函数调用CNN-LSTM模型
    '''
    mode = 'train'
    train_x,train_y = sliding_window(train)
    train_x = torch.tensor(train_x, dtype = torch.float32)
    train_y = torch.tensor(train_y, dtype = torch.float32)
    train_x = train_x.transpose(-1,-2)
    # torchkeras 训练方式
    model = torchkeras.Model(CNN_LSTM(window_size,fea_num))
    ds_train = TensorDataset(train_x, train_y)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0) 
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # 损失函数
    loss_fn = nn.MSELoss()
    model.compile(loss_func=loss_fn, optimizer=optimizer)
    model.fit(epochs=EPOCH,dl_train=dl_train)

    # # pytorch 训练方式
    # model = CNN_LSTM(window_size,fea_num)
    # # 损失函数
    # loss_fn = nn.MSELoss()
    # loss_seq = []
    # # 优化器
    # optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # # 开始训练
    # for epoch in range(EPOCH):
    #     loss_train = 0
    #     for _ in range(37):
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
    #     loss_seq.append(loss_train)
    #     print("[TRAIN] =========epoch : {},  loss  {:.4f}======== ".format(epoch+1,loss_train))
    return model

# # 训练小测试
# dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
#                    infer_datetime_format=True, engine='c',
#                    parse_dates=['datetime'], index_col=['datetime'])
# # 数据归一化处理，使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
# for i in range(dataset.shape[1]):
#     dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
# train,test = split_dataset(dataset)
# train_x, train_y = sliding_window(train)
# print(train_x.shape, train_y.shape)# (1099,7,7) (1099,7)
# cnn_model(train = train, window_size = 7, EPOCH = 100, lr = 0.3, batch_size = 30)

def forecast(model, input_x, sw_width):
    '''
    该函数实现对输入数据的预测
    '''
    # 重塑
    input_x = torch.tensor(input_x,dtype=torch.float32)
    # input_x = F.normalize(input_x)
    input_x = input_x.transpose(-1,-2)
    yhat = model(input_x) # 预测下周数据 verbose???
    yhat = yhat[0] # 获取预测向量
    return yhat

def evaluate_model(model, train, test, sd_width):
    '''
    该函数实现模型评估
    '''
    model.eval()
    test_x, test_y = sliding_window(test)
    history_fore = [x for x in test_x]
    history_fore = np.array(history_fore)
    predictions = list() # 用于保存每周的前向验证结果；
    with torch.no_grad():
        for i in range(test_x.shape[0]):
            input_x = history_fore[i, :] # 获取输入数据的最后一周的数据
            input_x = input_x.reshape([1,input_x.shape[0],input_x.shape[1]])
            yhat_sequence = forecast(model, input_x, sd_width) # 预测下周的数据
            predictions.append(yhat_sequence) # 保存预测结果
        predictions = np.array(predictions) # 评估一周中每天的预测结果
        # test_normalized = F.normalize(torch.tensor(test[:, :, 0], dtype=torch.float32))# 将测试集的数据归一化
        score, scores = evaluate_forecasts(test_y, predictions)
        print("Expected: {}\nPredictions: {}".format(test_y, predictions))
    return score, scores#, test_y, predictions

# dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
#                    infer_datetime_format=True, engine='c',
#                    parse_dates=['datetime'], index_col=['datetime'])
# for i in range(dataset.shape[1]):
#         dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
# train,test = split_dataset(dataset)
# model = cnn_model(train = train, lr = 0.001,EPOCH=50, batch_size=30, window_size=7)
# train_x, train_y = sliding_window(train)
# history_fore = [x for x in train_x]
# history_fore = np.array(history_fore)
# predictions = list() # 用于保存每周的前向验证结果；
# with torch.no_grad():
#     for i in range(train_x.shape[0]):
#         input_x = history_fore[i, :] # 获取输入数据的最后一周的数据
#         input_x = input_x.reshape([1,input_x.shape[0],input_x.shape[1]])
#         yhat_sequence = forecast(model, input_x, sw_width = 7) # 预测下周的数据
#         predictions.append(yhat_sequence) # 保存预测结果
#     predictions = np.array(predictions)

# train_y = train_y.reshape([-1])
# predictions = predictions.reshape([-1])

# plt.figure(figsize=(8,6), dpi=150)
# plt.plot([x for x in range(100)], train_y[0:100], label='test')
# plt.plot([x for x in range(100)], predictions[0:100], label='predictions')
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
    plt.title('CNN 模型预测结果',  size=18)
    plt.legend()
    plt.show()

def main_run(dataset, window_size, days, name, EPOCH, lr, fea_num, batch_size):
    '''
    主函数：数据处理、模型训练流程
    '''
    # 划分训练集和测试集
    train, test = split_dataset(dataset.values)
    # 训练模型
    model = cnn_model(train, EPOCH, lr, batch_size, window_size, fea_num)
    # 计算RMSE
    score, scores = evaluate_model(model, train, test, window_size)
    # 打印分数
    summarize_scores(name, score, scores)
    # 绘图
    model_plot(score, scores, days, name)

if __name__ == '__main__':
    
    dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
                   infer_datetime_format=True, engine='c',
                   parse_dates=['datetime'], index_col=['datetime'])
    # 数据归一化处理，使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
    for i in range(dataset.shape[1]):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))

    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    name = 'cnn'
    
    window_size = 7
    EPOCH = 20
    lr = 0.001
    fea_num = 7
    batch_size = 30
    
    main_run(dataset, window_size, days, name, EPOCH, lr, fea_num, batch_size)