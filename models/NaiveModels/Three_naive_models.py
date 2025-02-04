import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.metrics as skm
'''
有许多方法探索家庭用电量数据集。
本文使用这些数据来探索一个具体的问题：
用最近一段时间的数据来预测未来一周的用电量是多少。
这需要建立一个预测模型预测未来七天每天的总有功功率。
这类问题被称为多步时间序列预测问题。利用多个输入变量的模型可称为多变量（特征）多步时间序列预测模型。
'''
dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data',
                      header = 0,
                      infer_datetime_format = True,
                      engine = 'c',
                      parse_dates = ['datetime'],
                      index_col = ['datetime'])

'''
要达到这样的目的，为了便于处理，
先把原数据每分钟的耗电量采样数据重新采样成每日总耗电量。
这不是必需的，但是有意义，因为我们关心的是每天的总功率。可以使用 Pandas中的 resample() 函数实现，
设置参数 “D” 调用此函数，以允许按日期-时间索引加载数据并按天将数据重新采样分组。
然后，可以通过计算每天所有采样值的总和，为这8个变量（特征）创建新的日耗电量数据集。
'''

daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()#按天整理数据，进行合并
daily_data.to_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day')
'''
使用前三年的数据训练预测模型，最后一年的数据评估模型。将数据按照标准周（星期天开始到星期六结束）进行划分。
这对模型的选定是一种有效的方法，可以预测未来一周的耗电量。也有助于建模，模型可用于预测特定的一天（如周三）或整个序列。
以周为单位对数据进行处理，先截取测试数据，剩下的为训练数据。
数据的最后一年是2010年，2010年的第一个星期天是1月3日，数据在2010年11月26日结束，最近的最后一个星期六是11月20日。
一共46周的测试数据。下面提供测试数据集的第一行和最后一行每日数据以供确认。
'''
# print(daily_data.iloc[daily_data.index=='2010-01-03'])
# print(daily_data.iloc[daily_data.index=='2010-11-20'])
'''
每日耗电量数据从2006年12月16日开始。
数据集中的第一个周日是12月17日，这是第二行数据。将数据组织成标准周，可提供159个完整的标准周用于训练预测模型。
'''
'''
朴素预测模型（简单模型）
·每日持久性预测：模型取预测期前最后一天（如周六）的有功功率，作为预测期内（周日到周六）每天的有功功率
·每周持久性预测：模型取前一周作为下一周的预测，这是基于本周与前一周很相似的想法
·一年前每周的持久性预测：模型根据52周（一年）前的观察周进行预测
'''
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

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
    scores = list()# scores列表统计每日预测的 RMSE
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
    
    s = 0 # score统计总的 RMSE
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))
    return score, scores

def summarize_scores(name, score, scores):#?
    '''
    该函数用于统计误差
    name: 周几
    score: 整个数据集的RMSE
    scores: 每天的RMSE用列表存储
    '''
    s_scores = ', '.join(['%.4f' % s for s in scores])
    print('%s: [%.4f] %s\n' % (name, score, s_scores))

def evaluate_model(model_func, train, test):
    '''
    该函数实现评估单个模型
    '''
    history = [x for x in train] # # 以周为单位的数据列表，获得训练集内每一周的数据
    predictions = [] # 每周的前项预测值
    for i in range(len(test)):
        yhat_sequence = model_func(history) # 预测每周的耗电量
        predictions.append(yhat_sequence)
        history.append(test[i, :]) # 将测试数据中的采样值添加到history列表，以便预测下周的用电量
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions) # 评估一周中每天的预测损失，test[:,:,0]每一天的global_active_power
    return score, scores

def daily_persistence(history):
    '''
    该函数是用前一周最后一天的总有功功率作为预测期内（周日到周六）每天的有功功率
    '''
    last_week = history[-1] # 获取之前一周七天的总有功功率
    value = last_week[-1, 0] # 获取前一周最后一天的总有功功率
    forecast = [value for _ in range(7)] # 准备7天预测
    return forecast

def weekly_persistence(history):
    last_week = history[-1] # 将之前一周的数据作为预测数据
    return last_week[:, 0]

def week_one_year_ago_persistence(history):
    last_week = history[-52] # 将去年同一周的数据预测数据
    return last_week[:, 0]

def model_predict_plot(dataset, days):
    train, test = split_dataset(dataset.values)
    #定义要评估的模型的名称和函数
    models = dict()
    models['daily'] = daily_persistence
    models['weekly'] = weekly_persistence
    models['week-oya'] = week_one_year_ago_persistence
    
    plt.figure(figsize=(8,6), dpi=150)
    for name, func in models.items():
        score, scores = evaluate_model(func, train, test)
        summarize_scores(name, score, scores)
        plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('三种模型预测结果比较', color='blue', size=20)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data_Day', header=0, 
                       infer_datetime_format=True, engine='c',
                       parse_dates=['datetime'], index_col=['datetime'])
    # 对整个数据集进行归一化操作
    for i in range(dataset.shape[1]):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - min(dataset.iloc[:,i]))/(max(dataset.iloc[:,i]) - min(dataset.iloc[:,i]))
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    model_predict_plot(dataset, days)