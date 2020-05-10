# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:45:48 2019

@author: shaoqi
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'./光谱特征/原始/gouqi_spectrum_mean.csv', sep=',')
df = pd.read_csv(r'./光谱特征/SNV/gouqi_spectrum_mean_SNV.csv', sep=',')
df = pd.read_csv(r'./光谱特征/MSC/gouqi_spectrum_mean_MSC.csv', sep=',')
df = pd.read_csv(r'./光谱特征/SGFD/gouqi_spectrum_mean_SGFD.csv', sep=',')

x = df.drop(['label', 'DT', 'HT'], axis=1)
#y = df['HT']
y = df['DT']

corrs = []
for i in range(1, 96):
    corr = x[str(i)].corr(y) #计算相关系数
    corrs.append(corr)


figsize = 12, 9
figure, ax = plt.subplots(figsize=figsize)

############# 设置坐标刻度值的大小以及刻度值的字体 #############
plt.xlim(0, 96)
plt.ylim(-0.2, 0.3)
plt.xticks([i*10+5 for i in range(10)])
plt.tick_params(labelsize=22)

labels = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

labels = ax.get_xticklabels()
[label.set_fontname('Times New Roman') for label in labels]

############# 设置图例并且设置图例的字体及大小 #############
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 25,
         }

plt.plot([i+1 for i in range(95)], corrs, color="grey", lw=2, ls = "dashed")

plt.scatter([i+1 for i in range(95)], corrs, s=90, marker="d", color="dodgerblue", alpha=0.9)

plt.savefig('SGFD_coef_DT.eps', dpi=2000)

plt.show()