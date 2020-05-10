# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:45:48 2019

@author: shaoqi
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def WZ_result(X1, y1, X, y, wz):

    #X = X.values
    y = y.values
    rmses=[]
    rf=[]
    
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        
        train_X, test_X, train_y, test_y = X[train],X[test],y[train],y[test]
        clf = MLPRegressor(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(80, 40))
        clf.fit(train_X,train_y)
        predicted = clf.predict(test_X)
        
        rmse = mean_squared_error(test_y, predicted) ** 0.5
        rmses.append(rmse)
        
        rf.append(clf)
    
    index = rmses.index(min(rmses))
    predict = rf[index].predict(X1)
    
    # rmse  mae r2
    r2 = r2_score(y1, predict)
    print("R2",r2)
    
    mae = mean_absolute_error(y1, predict)
    print("mae", mae)
    
    rmse = mean_squared_error(y1, predict) ** 0.5
    print("rmse", rmse)
    
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    
    p0, = plt.plot([0,8], [0,8], '--', color='black', label='line', linewidth = 0.8, zorder=10)
    
    color = ['limegreen', 'mediumslateblue', 'dodgerblue', 'darkorange']
    marker = ['X', 'o', 'd', '<']
    
    p1 = plt.scatter(y1[0:12], predict[0:12], c=color[0], marker=marker[0], label='NQ1', s=180, alpha=0.75, zorder=20)
    p2 = plt.scatter(y1[12:24], predict[12:24], c=color[1], marker=marker[1], label='NQ5', s=180,alpha=0.75, zorder=20)
    p3 = plt.scatter(y1[24:36], predict[24:36], c=color[2], marker=marker[2], label='NQ7', s=180, alpha=0.75, zorder=20)
    p4 = plt.scatter(y1[36:48], predict[36:48], c=color[3], marker=marker[3], label='NNQ9', s=180, alpha=0.75, zorder=20)
    
    
    ############# 设置坐标刻度值的大小以及刻度值的字体 #############
    #plt.xlim(3, 7.5)
    #plt.ylim(3, 7.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=18)
    
    labels = ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    labels = ax.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    ############# 设置图例并且设置图例的字体及大小 #############
    font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 25,
         }

    plt.legend(prop=font1, frameon=False)  # 图例
    plt.ylabel('      (g / 100g)', font1)
    plt.xlabel('      (g / 100g)', font1)
    
    ax = plt.gca()
    ax.set_aspect(1)
    
    #plt.savefig('C:\\Users\\shaoqi\\Desktop\\'+ wz +'.eps', dpi=2000)
    plt.show()
    return predict
    

df = pd.read_csv(r'./全波段光谱+纹理/gouqi_spectrum_texture_none.csv', sep=',')
df = pd.read_csv(r'./全波段光谱+纹理/gouqi_spectrum_texture_SNV.csv', sep=',')
#df = pd.read_csv(r'./全波段光谱+纹理/gouqi_spectrum_texture_MSC.csv', sep=',')
#df = pd.read_csv(r'./全波段光谱+纹理/gouqi_spectrum_texture_SGFD.csv', sep=',')


#df = pd.read_csv(r'./光谱特征/原始/gouqi_spectrum_mean.csv', sep=',')
#df = pd.read_csv(r'./光谱特征/SNV/gouqi_spectrum_mean_SNV.csv', sep=',')
#df = pd.read_csv(r'./光谱特征/MSC/gouqi_spectrum_mean_MSC.csv', sep=',')
#df = pd.read_csv(r'./光谱特征/SGFD/gouqi_spectrum_mean_SGFD.csv', sep=',')


scale = preprocessing.StandardScaler()

X1 = df.drop(['label', 'DT', 'HT'], axis=1)
y1 = df['HT']
scale_fit = scale.fit(X1)
X1 = scale_fit.transform(X1)

dff = shuffle(df)

X = dff.drop(['label', 'DT', 'HT'], axis=1)
y = dff['HT']
X = scale_fit.transform(X)

predict = WZ_result(X1, y1, X, y, 'None_DT_NN')

df_result = pd.DataFrame()
df_result['predict'] = predict
df_result['HT'] = y1
df_result['label'] = df['label']
df_result.to_csv("Coef_NN_HT_MSC.csv", index=False)

