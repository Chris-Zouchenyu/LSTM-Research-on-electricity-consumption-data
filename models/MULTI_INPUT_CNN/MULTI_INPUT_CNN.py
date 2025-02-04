import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CNN_model import *
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from torch.utils.data import TensorDataset,DataLoader
import torchkeras
torch.manual_seed(1234)

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
    s_scores = ', '.join(['%.4f' % s for s in scores])
    print('%s: [%.4f] %s\n' % (name, score, s_scores))
    
def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    截取所有特征
    '''
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2])) # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []
    
    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out
        
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 因为是for循环，所以滑动窗口的滑动步长为1；想调整滑动步长可以通过yield实现，后边的文章会讲；
            X.append(data[in_start:in_end, :]) # 截取窗口宽度数量的采样点的全部8个特征
            y.append(data[in_end:out_end, 0]) # 截取样本之后7个时间步长的总有功功耗（截取一个单列片段，有7个元素）
        in_start += 1
        
    return np.array(X), np.array(y)

def multi_input_cnn_model(train, sw_width, in_start=0, verbose=0, epochs=20, batch_size=16):
    '''
    该函数定义 多输入序列 CNN 模型
    '''
    train_x, train_y = sliding_window(train, sw_width, in_start=0)
    train_x = torch.tensor(train_x, dtype = torch.float32)
    train_y = torch.tensor(train_y, dtype = torch.float32)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_x = train_x.transpose(-1,-2)
    # torchkeras训练
    model = torchkeras.Model(CNN(n_timesteps, n_features, n_outputs))
    ds_train = TensorDataset(train_x, train_y)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0) 
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # 损失函数
    loss_fn = nn.MSELoss()
    model.compile(loss_func=loss_fn, optimizer=optimizer)
    model.fit(epochs=epochs_num,dl_train=dl_train)

    # pytorch训练
    # model = CNN(n_timesteps, n_features, n_outputs)
    # # 优化器
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    # # 损失函数
    # loss_fn = nn.MSELoss()
    # loss_seq = []
    # # 开始训练
    # for epoch in range(epochs):
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
    #     torch.save(model,'nn_MULTI_INPUT_CNN.pth')
    return model



# # 数据归一化处理，使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
# for i in range(dataset.shape[1]):
#     dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
# train,test = split_dataset(dataset)
# train_x, train_y = sliding_window(train)
# print(train_x.shape, train_y.shape)
# multi_input_cnn_model(train, sw_width = 7, in_start=0, verbose=0, epochs=20, batch_size=16)


def forecast(model, pred_seq, sw_width):
    '''
    该函数实现对输入数据的预测
    多个特征
    '''
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
    input_x = data[-sw_width:, :] # 获取输入数据的最后一周的数据
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1])) # 重塑形状为[1, sw_width, n]
    input_x = torch.tensor(input_x, dtype = torch.float32)
    input_x = input_x.transpose(-1,-2)
    
    model = torch.load('nn_MULTI_INPUT_CNN.pth')

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
    plt.title('多输入序列 CNN 模型预测结果',  size=18)
    plt.legend()
    plt.show()
    
def main_run(dataset, sw_width, days, name, in_start, verbose, epochs, batch_size):
    '''
    主函数：数据处理、模型训练流程
    '''
    # 划分训练集和测试集
    train, test = split_dataset(dataset.values)
    # 训练模型
    model = multi_input_cnn_model(train, sw_width, in_start, verbose, epochs, batch_size)# 没保存模型嘛？？？？
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
    for i in range(dataset.shape[1]):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))

    
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    name = 'cnn'
    
    sliding_window_width=7
    input_sequence_start=0
    
    epochs_num=20
    batch_size_set=16
    verbose_set=0
    
    
    main_run(dataset, sliding_window_width, days, name, input_sequence_start,
             verbose_set, epochs_num, batch_size_set)
