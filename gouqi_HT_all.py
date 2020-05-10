# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:12:04 2020

@author: shaoqi
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import pandas as pd
import matplotlib.pyplot as plt

index = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]]

## SPA_T
file = [["SPA_T_RF_HT_None.csv", "SPA_T_NN_HT_None.csv", "SPA_T_SVM_HT_None.csv"],
        ["SPA_T_RF_HT_SNV.csv", "SPA_T_NN_HT_SNV.csv", "SPA_T_SVM_HT_SNV.csv"],
        ["SPA_T_RF_HT_MSC.csv", "SPA_T_NN_HT_MSC.csv", "SPA_T_SVM_HT_MSC.csv"],
        ["SPA_T_RF_HT_SGFD.csv", "SPA_T_NN_HT_SGFD.csv", "SPA_T_SVM_HT_SGFD.csv"]]

r2 = [['0.9135', '0.8196', '0.8873'],
      ['0.9223', '0.8561', '0.8998'],
      ['0.9350', '0.8815', '0.9177'],
      ['0.9426', '0.8932', '0.9214']]

mae = [['0.0248', '0.0470', '0.0357'],
       ['0.0264', '0.0432', '0.0335'],
       ['0.0257', '0.0361', '0.0307'],
       ['0.0251', '0.0365', '0.0306']]

rmse = [['0.0418', '0.0603', '0.0477'],
        ['0.0396', '0.0538', '0.0449'],
        ['0.0362', '0.0489', '0.0407'],
        ['0.0340', '0.0464', '0.0398']]
'''


## coef_T
file = [["Coef_T_RF_HT_None.csv", "Coef_T_NN_HT_None.csv", "Coef_T_SVM_HT_None.csv"],
        ["Coef_T_RF_HT_SNV.csv", "Coef_T_NN_HT_SNV.csv", "Coef_T_SVM_HT_SNV.csv"],
        ["Coef_T_RF_HT_MSC.csv", "Coef_T_NN_HT_MSC.csv", "Coef_T_SVM_HT_MSC.csv"],
        ["Coef_T_RF_HT_SGFD.csv", "Coef_T_NN_HT_SGFD.csv", "Coef_T_SVM_HT_SGFD.csv"]]

r2 = [['0.8982', '0.8014', '0.8693'],
      ['0.9160', '0.8454', '0.8956'],
      ['0.9235', '0.8505', '0.9055'],
      ['0.9399', '0.8873', '0.9144']]

mae = [['0.0284', '0.0516', '0.0389'],
      ['0.0278', '0.0432', '0.0347'],
      ['0.0279', '0.0417', '0.0318'],
      ['0.0269', '0.0392', '0.0311']]

rmse = [['0.0453', '0.0633', '0.0514'],
      ['0.0412', '0.0559', '0.0459'],
      ['0.0393', '0.0549', '0.0436'],
      ['0.0348', '0.0477', '0.0416']]



## PCA_T
file = [["PCA_T_RF_HT_None.csv", "PCA_T_SVM_HT_None.csv", "PCA_T_NN_HT_None.csv"],
        ["PCA_T_RF_HT_SNV.csv", "PCA_T_SVM_HT_SNV.csv", "PCA_T_NN_HT_SNV.csv"],
        ["PCA_T_RF_HT_MSC.csv", "PCA_T_SVM_HT_MSC.csv", "PCA_T_NN_HT_MSC.csv"],
        ["PCA_T_RF_HT_SGFD.csv", "PCA_T_SVM_HT_SGFD.csv", "PCA_T_NN_HT_SGFD.csv"]]

r2 = [['0.8848', '0.8108', '0.8053'],
      ['0.9129', '0.8643', '0.8259'],
      ['0.9121', '0.8929', '0.8398'],
      ['0.9185', '0.9061', '0.8674']]

mae = [['0.0342', '0.0474', '0.0521'],
      ['0.0265', '0.0408', '0.0469'],
      ['0.0276', '0.0342', '0.0448'],
      ['0.0253', '0.0321', '0.0399']]

rmse = [['0.0482', '0.0618', '0.0627'],
      ['0.0419', '0.0523', '0.0593'],
      ['0.0421', '0.0465', '0.0568'],
      ['0.0405', '0.0435', '0.0517']]



## PCA
file = [["PCA_RF_HT_None.csv", "PCA_SVM_HT_None.csv", "PCA_NN_HT_None.csv"],
        ["PCA_RF_HT_SNV.csv", "PCA_NN_HT_SNV.csv", "PCA_SVM_HT_SNV.csv"],
        ["PCA_RF_HT_MSC.csv", "PCA_SVM_HT_MSC.csv", "PCA_NN_HT_MSC.csv"],
        ["PCA_RF_HT_SGFD.csv", "PCA_NN_HT_SGFD.csv", "PCA_SVM_HT_SGFD.csv"]]

r2 = [['0.8672', '0.7703', '0.7724'],
      ['0.8908', '0.8025', '0.8036'],
      ['0.8953', '0.8362', '0.8475'],
      ['0.9138', '0.8714', '0.8744']]

mae = [['0.0309', '0.0591', '0.0539'],
      ['0.0298', '0.0486', '0.0517'],
      ['0.0303', '0.0435', '0.0438'],
      ['0.0307', '0.0399', '0.0387']]

rmse = [['0.0517', '0.0681', '0.0678'],
      ['0.0469', '0.0631', '0.0629'],
      ['0.0459', '0.0575', '0.0555'],
      ['0.0417', '0.0509', '0.0503']]


## coef
file = [["Coef_RF_HT_None.csv", "Coef_NN_HT_None.csv", "Coef_SVM_HT_None.csv"],
        ["Coef_RF_HT_SNV.csv", "Coef_NN_HT_SNV.csv", "Coef_SVM_HT_SNV.csv"],
        ["Coef_RF_HT_MSC.csv", "Coef_NN_HT_SGFD.csv", "Coef_SVM_HT_MSC.csv"],
        ["Coef_RF_HT_SGFD.csv", "Coef_NN_HT_MSC.csv", "Coef_SVM_HT_SGFD.csv"]]

r2 = [['0.8836', '0.7695', '0.8167'],
      ['0.8974', '0.8310', '0.8658'],
      ['0.9062', '0.8698', '0.8979'],
      ['0.9158', '0.8824', '0.8988']]

mae = [['0.0311', '0.0541', '0.0498'],
      ['0.0286', '0.0475', '0.0387'],
      ['0.0299', '0.0406', '0.0339'],
      ['0.0274', '0.0388', '0.0354']]

rmse = [['0.0485', '0.0682', '0.0608'],
      ['0.0455', '0.0584', '0.0521'],
      ['0.0435', '0.0512', '0.0453'],
      ['0.0412', '0.0487', '0.0452']]


## SPA
file = [["SPA_RF_HT_None.csv", "SPA_NN_HT_None.csv", "SPA_SVM_HT_None.csv"],
        ["SPA_RF_HT_SNV.csv", "SPA_NN_HT_SNV.csv", "SPA_SVM_HT_SNV.csv"],
        ["SPA_RF_HT_MSC.csv", "SPA_NN_HT_MSC.csv", "SPA_SVM_HT_MSC.csv"],
        ["SPA_RF_HT_SGFD.csv", "SPA_NN_HT_SGFD.csv", "SPA_SVM_HT_SGFD.csv"]]

r2 = [['0.8936', '0.8053', '0.8687'],
      ['0.9168', '0.8513', '0.8700'],
      ['0.9182', '0.8570', '0.9075'],
      ['0.9306', '0.8858', '0.9007']]

mae = [['0.0342', '0.0512', '0.0385'],
      ['0.0288', '0.0404', '0.0394'],
      ['0.0269', '0.0419', '0.0316'],
      ['0.0249', '0.0375', '0.0335']]

rmse = [['0.0463', '0.0627', '0.0515'],
      ['0.0409', '0.0548', '0.0512'],
      ['0.0406', '0.0537', '0.0432'],
      ['0.0374', '0.0479', '0.0447']]
  
### FB
file = [["FB_RF_HT_None.csv", "FB_NN_HT_None.csv", "FB_SVM_HT_None.csv"],
        ["FB_RF_HT_SNV.csv", "FB_NN_HT_SNV.csv", "FB_SVM_HT_SNV.csv"],
        ["FB_RF_HT_MSC.csv", "FB_NN_HT_MSC.csv", "FB_SVM_HT_MSC.csv"],
        ["FB_RF_HT_SGFD.csv", "FB_NN_HT_SGFD.csv", "FB_SVM_HT_SGFD.csv"]]

r2 = [['0.9025', '0.8208', '0.8365'],
      ['0.9272', '0.8223', '0.8775'],
      ['0.9346', '0.8481', '0.9014'],
      ['0.9442', '0.9003', '0.9157']]

mae = [['0.0276', '0.0455', '0.0451'],
      ['0.0272', '0.0461', '0.0392'],
      ['0.0271', '0.0423', '0.0349'],
      ['0.0257', '0.0335', '0.0307']]

rmse = [['0.0443', '0.0601', '0.0574'],
      ['0.0383', '0.0599', '0.0497'],
      ['0.0363', '0.0554', '0.0446'],
      ['0.0335', '0.0448', '0.0412']]

## FB_T
file = [["FB_T_RF_HT_None.csv", "FB_T_NN_HT_None.csv", "FB_T_SVM_HT_None.csv"],
        ["FB_T_RF_HT_SNV.csv", "FB_T_NN_HT_SNV.csv", "FB_T_SVM_HT_SNV.csv"],
        ["FB_T_RF_HT_MSC.csv",  "FB_T_NN_HT_MSC.csv", "FB_T_SVM_HT_MSC.csv"],
        ["FB_T_RF_HT_SGFD.csv", "FB_T_NN_HT_SGFD.csv", "FB_T_SVM_HT_SGFD.csv"]]

r2 = [['0.9119', '0.8644', '0.8799'],
      ['0.9406', '0.8887', '0.8996'],
      ['0.9458', '0.9088', '0.9135'],
      ['0.9559', '0.9158', '0.9395']]

mae = [['0.0314', '0.0397', '0.0366'],
      ['0.0227', '0.0374', '0.0326'],
      ['0.0257', '0.0314', '0.0319'],
      ['0.0207', '0.0316', '0.0262']]

rmse = [['0.0422', '0.0523', '0.0492'],
      ['0.0346', '0.0474', '0.0450'],
      ['0.0331', '0.0429', '0.0418'],
      ['0.0298', '0.0412', '0.0349']]


'''

figsize = 30,35
plt.figure(figsize=figsize)

for i in range(4):
    for j in range(3):
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/全波段/"+ file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/全波段纹理/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/PCA/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/回归系数/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/SPA/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/PCA纹理/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/回归系数纹理/" + file[i][j], encoding='utf-8')
        df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/黄酮/SPA纹理/" + file[i][j], encoding='utf-8')

        ax = plt.subplot(4, 3, index[i][j])
        
        y1 = df['HT'].to_list()
        predict = df['predict'].to_list()

        '''
        # rmse  mae r2
        r2 = r2_score(y1, predict)
        print("R2",r2)
        
        mae = mean_absolute_error(y1, predict)
        print("mae", mae)
        
        rmse = mean_squared_error(y1, predict) ** 0.5
        print("rmse", rmse)

        '''


        color = ['limegreen', 'mediumslateblue', 'dodgerblue', 'darkorange']
        marker = ['X', 'o', 'd', '<']
        
        p1 = plt.scatter(y1[0:12], predict[0:12], c=color[0], marker=marker[0], label='', s=180, alpha=0.7, zorder=20)
        p2 = plt.scatter(y1[12:24], predict[12:24], c=color[1], marker=marker[1], label='', s=180,alpha=0.7, zorder=20)
        p3 = plt.scatter(y1[24:36], predict[24:36], c=color[2], marker=marker[2], label='', s=180, alpha=0.7, zorder=20)
        p4 = plt.scatter(y1[36:48], predict[36:48], c=color[3], marker=marker[3], label='', s=180, alpha=0.7, zorder=20)
        
        p0, = plt.plot([0,8], [0,8], '--', color='black', label='',linewidth = 1.0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='$\mathregular{R^2}$ = '+r2[i][j],linewidth = 0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='RMSE = ' + rmse[i][j],linewidth = 0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='MAE = ' + mae[i][j],linewidth = 0)
        
        ############# 设置坐标刻度值的大小以及刻度值的字体 #############
        plt.xlim(0.2, 0.8)
        plt.ylim(0.2, 0.8)
        plt.xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        plt.tick_params(labelsize=25)
        
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


plt.savefig('SPA_T_HT.eps', dpi=2000)
plt.show()
