import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\OriginalData\household_power_consumption.txt',
                      sep = ';',
                      header = 0,
                      low_memory = False,
                      infer_datetime_format = True,
                      engine = 'c',#使用解析器的引擎，C引擎更快
                      parse_dates = {'datetime':[0,1]},#将原数据中的第一二列作为新的列名，命名为‘datatime’列
                      index_col = ['datetime']
                      )
# print(dataset.shape)
# print(dataset.head(10))#查看前十行数据
# print(dataset.isna().sum())#查看缺失值
# print(dataset.iloc[dataset.values == '?'].count())#查看有默认标记的异常值
dataset.replace('?',np.nan,inplace = True)#将有标记的异常值用nan替换
values = dataset.values.astype('float32')
#缺失值处理
def fill_missing(values):
    '''
    函数用于填充缺失值，具体方法是用前一天同时刻的数据进行填充
    '''
    one_day = 60*24
    for row in range(values.shape[0]):#行循环
        for col in range(values.shape[1]):#列循环
            if np.isnan(values[row,col]):
                values[row,col] = values[row-one_day,col]
fill_missing(values)
# print(np.sum(np.isnan(values)))#查看缺失值 为0

#添加一列，原数据（sub_metering1,2,3）只提供了三个来源的有功电能，并没有提供其他来源的有功电能，这里用总电能减去三个来源的有功电能，得到其他来源电能损耗，存到新的一列
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
dataset.to_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data')


