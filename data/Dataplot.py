import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\python\Deep learning\法国家庭用电量预测项目\Processed_data',
                      header = 0,
                      infer_datetime_format = True,
                      engine = 'c',
                      parse_dates = ['datetime'],
                      index_col = ['datetime'])
#创建一个八子图的图像，每个子图对应一个变量
def plot_features(dataset):
    plt.figure(figsize = (16,12), dpi = 300)
    for i in range(len(dataset.columns)):#dataset.columns是列名的意思
        plt.subplot(len(dataset.columns),1,i+1)#i+1索引值
        feature_name = dataset.columns[i]
        plt.plot(dataset[feature_name])
        plt.title(feature_name, y = 0)
        plt.grid()
    
    plt.tight_layout()#调整距离，防止字重叠
    plt.show()
#为每年创建一个有功功率图 这个没问题 改好了
def plot_year_gap(dataset, years_list):
    plt.figure(figsize=(16,12), dpi=150)
    for i in range(len(years_list)):
        year = years_list[i]
        start_date = f'{year}-01-01'
        end_date = f'{year+1}-01-01'
        year_data = dataset.loc[start_date:end_date]

        ax = plt.subplot(len(years_list), 1, i+1)
        ax.set_ylabel(r'$KW$')

        plt.plot(year_data['Global_active_power'])
        plt.title(str(year), y=0, loc='left')
    plt.tight_layout()
    plt.show()

#进一步查看每个月的用电情况。比如查看2008年每个月的有功功率，可能有助于梳理出几个月的变化规律，如每日和每周用电状况规律
def plot_month_gap(dataset, year, months_list):
    plt.figure(figsize=(16,12), dpi=150)
    for i in range(len(months_list)):
        month = months_list[i]
        start_date = f'{year}-{month:02d}-01'
        if month==12:
            end_date = f'{year+1}-01-01'
        else:
            end_date = f'{year}-{month+1:02d}-01'
        
        ax = plt.subplot(len(months_list), 1, i+1)
        ax.set_ylabel(r'$KW$')

        month_data = dataset.loc[start_date:end_date]

        
        plt.plot(month_data['Global_active_power'])
        plt.title(f'{year}-{month:02d}', y=0, loc='left')
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
#进一步查看每日的用电情况
def plot_day_gap(dataset, year, month, days_list):
    plt.figure(figsize=(20,24), dpi=150)
    for i in range(len(days_list)):
        day = days_list[i]
        start_date = f'{year}-{month:02d}-{day:02d} 00:00:00'
        end_date = f'{year}-{month:02d}-{day+1:02d} 00:00:00'
        day_data = dataset.loc[start_date:end_date]

        ax = plt.subplot(len(days_list), 1, i+1)
        ax.set_ylabel(r'$KW$',size=6)
        
        gcp_data = day_data['Global_active_power']
        plt.plot(gcp_data)
        plt.title(day, y=0, loc='left', size=6)
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(rotation=0)

    plt.show()
# 时间序列数据分布
def dataset_distribution(dataset):
    plt.figure(figsize=(16,12), dpi=150)
    for i in range(len(dataset.columns)):
        ax = plt.subplot(len(dataset.columns), 1, i+1)
        ax.set_ylabel(r'$numbers$',size=10)
        feature_name = dataset.columns[i]
        dataset[feature_name].hist(bins=100)
        plt.title(feature_name, y=0, loc='right', size = 10)
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(rotation=0)
        
    plt.tight_layout()
    plt.show()
'''
有功和无功功率、强度以及分表功率都是向瓦时或千瓦倾斜的分布，
电压数据呈高斯分布
有功功率的分布似乎是双峰的，这意味着它看起来有两组观测值。可以通过查看四年来的数据的有功功率分布来验证
'''
def plot_year_dist(dataset, years_list):
    plt.figure(figsize=(16,12), dpi=150)
    for i in range(len(years_list)):
        year = years_list[i]
        start_date = f'{year}-01-01'
        end_date = f'{year+1}-01-01'
        year_data = dataset.loc[start_date:end_date]

        ax = plt.subplot(len(years_list), 1, i+1)
        ax.set_ylabel(r'$numbers$')
        ax.set_xlim(0, 5) # 设置x轴显示限制，保证每个子图x刻度相同
        year_data['Global_active_power'].hist(bins=100, histtype='bar')
        
        plt.title(str(year), y=0, loc='right',size = 10)
        plt.xticks(rotation = 0)
    plt.tight_layout()
    plt.show()
#可以看到，有功功率分布看起来非常相似。这种分布确实是双峰的，一个峰值约为0.3kw，另一个峰值约为1.3kw。随着有功功率（x轴）的增加，高功率用电时间点的数量越来越少










