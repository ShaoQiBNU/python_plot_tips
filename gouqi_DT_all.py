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

## SPA
file = [["SPA_RF_DT_None.csv", "SPA_SVM_DT_None.csv", "SPA_NN_DT_None.csv"],
        ["SPA_RF_DT_SNV.csv", "SPA_SVM_DT_SNV.csv", "SPA_NN_DT_SNV.csv"],
        ["SPA_RF_DT_MSC.csv", "SPA_SVM_DT_MSC.csv", "SPA_NN_DT_MSC.csv"],
        ["SPA_RF_DT_SGFD.csv", "SPA_SVM_DT_SGFD.csv", "SPA_NN_DT_SGFD.csv"]]

r2 = [['0.7655', '0.7316', '0.7458'],
      ['0.8372', '0.8178', '0.8256'],
      ['0.8556', '0.8193', '0.8525'],
      ['0.8687', '0.8249', '0.8479']]

mae = [['0.3442', '0.3631', '0.4047'],
      ['0.2941', '0.3104', '0.2994'],
      ['0.2683', '0.3078', '0.2884'],
      ['0.2746', '0.2813', '0.2984']]

rmse = [['0.4645', '0.4968', '0.4835'],
      ['0.3869', '0.4093', '0.4005'],
      ['0.3644', '0.4077', '0.3684'],
      ['0.3475', '0.4013', '0.3741']]



'''
## SPA_T
file = [["SPA_T_RF_DT_None.csv", "SPA_T_SVM_DT_None.csv", "SPA_T_NN_DT_None.csv"],
        ["SPA_T_RF_DT_SNV.csv", "SPA_T_SVM_DT_SNV.csv", "SPA_T_NN_DT_SNV.csv"],
        ["SPA_T_RF_DT_MSC.csv", "SPA_T_SVM_DT_MSC.csv", "SPA_T_NN_DT_MSC.csv"],
        ["SPA_T_RF_DT_SGFD.csv", "SPA_T_SVM_DT_SGFD.csv", "SPA_T_NN_DT_SGFD.csv"]]

r2 = [['0.8062', '0.7909', '0.7606'],
      ['0.8439', '0.8201', '0.8520'],
      ['0.8570', '0.8213', '0.8692'],
      ['0.8843', '0.8299', '0.8656']]

mae = [['0.3305', '0.3089', '0.3683'],
      ['0.2961', '0.2906', '0.2975'],
      ['0.2859', '0.3038', '0.2927'],
      ['0.2637', '0.3211', '0.2692']]

rmse = [['0.4222', '0.4386', '0.4692'],
      ['0.3789', '0.4068', '0.3689'],
      ['0.3627', '0.4054', '0.3469'],
      ['0.3262', '0.3955', '0.3516']]


## coef_T
file = [["Coef_T_RF_DT_None.csv", "Coef_T_SVM_DT_None.csv", "Coef_T_NN_DT_None.csv"],
        ["Coef_T_RF_DT_SNV.csv", "Coef_T_SVM_DT_SNV.csv", "Coef_T_NN_DT_SNV.csv"],
        ["Coef_T_RF_DT_MSC.csv", "Coef_T_SVM_DT_MSC.csv", "Coef_T_NN_DT_MSC.csv"],
        ["Coef_T_RF_DT_SGFD.csv", "Coef_T_SVM_DT_SGFD.csv", "Coef_T_NN_DT_SGFD.csv"]]

r2 = [['0.7693', '0.7629', '0.7712'],
      ['0.8541', '0.8017', '0.8306'],
      ['0.8471', '0.8149', '0.8432'],
      ['0.8506', '0.8229', '0.8400']]

mae = [['0.3328', '0.3474', '0.3505'],
      ['0.2697', '0.2789', '0.2844'],
      ['0.2661', '0.3270', '0.3037'],
      ['0.2892', '0.3095', '0.3127']]

rmse = [['0.4606', '0.4669', '0.4588'],
      ['0.3663', '0.4270', '0.3947'],
      ['0.3750', '0.4126', '0.3798'],
      ['0.3707', '0.4036', '0.3836']]


## PCA_T
file = [["PCA_T_RF_DT_None.csv", "PCA_T_SVM_DT_None.csv", "PCA_T_NN_DT_None.csv"],
        ["PCA_T_RF_DT_SNV.csv", "PCA_T_SVM_DT_SNV.csv", "PCA_T_NN_DT_SNV.csv"],
        ["PCA_T_RF_DT_MSC.csv", "PCA_T_SVM_DT_MSC.csv", "PCA_T_NN_DT_MSC.csv"],
        ["PCA_T_RF_DT_SGFD.csv", "PCA_T_SVM_DT_SGFD.csv", "PCA_T_NN_DT_SGFD.csv"]]

r2 = [['0.7612', '0.7696', '0.7897'],
      ['0.8207', '0.8022', '0.8059'],
      ['0.8419', '0.8258', '0.8205'],
      ['0.8588', '0.8328', '0.8465']]

mae = [['0.3352', '0.3460', '0.3371'],
      ['0.2973', '0.3343', '0.3205'],
      ['0.2911', '0.3110', '0.3131'],
      ['0.2802', '0.3005', '0.3017']]

rmse = [['0.4686', '0.4604', '0.4398'],
      ['0.4061', '0.4266', '0.4226'],
      ['0.3813', '0.4003', '0.4062'],
      ['0.3604', '0.3922', '0.3757']]
## SPA
file = [["SPA_RF_DT_None.csv", "SPA_SVM_DT_None.csv", "SPA_NN_DT_None.csv"],
        ["SPA_RF_DT_SNV.csv", "SPA_SVM_DT_SNV.csv", "SPA_NN_DT_SNV.csv"],
        ["SPA_RF_DT_MSC.csv", "SPA_SVM_DT_MSC.csv", "SPA_NN_DT_MSC.csv"],
        ["SPA_RF_DT_SGFD.csv", "SPA_SVM_DT_SGFD.csv", "SPA_NN_DT_SGFD.csv"]]

r2 = [['0.7655', '0.7316', '0.7458'],
      ['0.8372', '0.8178', '0.8256'],
      ['0.8556', '0.8193', '0.8525'],
      ['0.8687', '0.8249', '0.8479']]

mae = [['0.3442', '0.3631', '0.4047'],
      ['0.2941', '0.3104', '0.2994'],
      ['0.2683', '0.3078', '0.2884'],
      ['0.2746', '0.2813', '0.2984']]

rmse = [['0.4645', '0.4968', '0.4835'],
      ['0.3869', '0.4093', '0.4005'],
      ['0.3644', '0.4077', '0.3684'],
      ['0.3475', '0.4013', '0.3741']]


## coef
file = [["Coef_RF_DT_None.csv", "Coef_SVM_DT_None.csv", "Coef_NN_DT_None.csv"],
        ["Coef_RF_DT_SNV.csv", "Coef_SVM_DT_SNV.csv", "Coef_NN_DT_SNV.csv"],
        ["Coef_RF_DT_MSC.csv", "Coef_SVM_DT_MSC.csv", "Coef_NN_DT_MSC.csv"],
        ["Coef_RF_DT_SGFD.csv", "Coef_SVM_DT_SGFD.csv", "Coef_NN_DT_SGFD.csv"]]

r2 = [['0.7489', '0.7428', '0.7334'],
      ['0.8338', '0.7983', '0.8263'],
      ['0.8399', '0.8099', '0.8331'],
      ['0.8369', '0.8163', '0.8345']]

mae = [['0.3440', '0.3492', '0.3929'],
      ['0.2747', '0.3149', '0.2909'],
      ['0.3048', '0.3229', '0.2972'],
      ['0.2945', '0.3135', '0.2886']]

rmse = [['0.4805', '0.4863', '0.4953'],
      ['0.3910', '0.4307', '0.3997'],
      ['0.3838', '0.4182', '0.3919'],
      ['0.3873', '0.4110', '0.3901']]
      
## PCA
file = [["PCA_RF_DT_None.csv", "PCA_SVM_DT_None.csv", "PCA_NN_DT_None.csv"],
        ["PCA_RF_DT_SNV.csv", "PCA_SVM_DT_SNV.csv", "PCA_NN_DT_SNV.csv"],
        ["PCA_RF_DT_MSC.csv", "PCA_SVM_DT_MSC.csv", "PCA_NN_DT_MSC.csv"],
        ["PCA_RF_DT_SGFD.csv", "PCA_SVM_DT_SGFD.csv", "PCA_NN_DT_SGFD.csv"]]

r2 = [['0.7280', '0.7245', '0.7412'],
      ['0.8253', '0.7828', '0.7829'],
      ['0.8307', '0.7651', '0.8132'],
      ['0.8457', '0.7983', '0.8318']]

mae = [['0.3808', '0.3779', '0.3799'],
      ['0.3018', '0.2884', '0.3333'],
      ['0.3065', '0.3281', '0.3036'],
      ['0.2898', '0.2993', '0.3288']]

rmse = [['0.5002', '0.5033', '0.4879'],
      ['0.4009', '0.4469', '0.4469'],
      ['0.3946', '0.4648', '0.4146'],
      ['0.3767', '0.4307', '0.3934']]

## FB_T
file = [["FB_T_RF_DT_None.csv", "FB_T_SVM_DT_None.csv", "FB_T_NN_DT_None.csv"],
        ["FB_T_RF_DT_SNV.csv", "FB_T_SVM_DT_SNV.csv", "FB_T_NN_DT_SNV.csv"],
        ["FB_T_RF_DT_MSC.csv", "FB_T_SVM_DT_MSC.csv", "FB_T_NN_DT_MSC.csv"],
        ["FB_T_RF_DT_SGFD.csv", "FB_T_SVM_DT_SGFD.csv", "FB_T_NN_DT_SGFD.csv"]]

r2 = [['0.8377', '0.7647', '0.8151'],
      ['0.8677', '0.8212', '0.8545'],
      ['0.8744', '0.8289', '0.8799'],
      ['0.8989', '0.8386', '0.8724']]

mae = [['0.2693', '0.3478', '0.3323'],
      ['0.2668', '0.3143', '0.2673'],
      ['0.2792', '0.2930', '0.2708'],
      ['0.2196', '0.3049', '0.2641']]

rmse = [['0.3863', '0.4652', '0.4124'],
      ['0.3488', '0.4056', '0.3658'],
      ['0.3398', '0.3967', '0.3322'],
      ['0.3049', '0.3853', '0.3425']]

### FB
file = [["FB_RF_DT_None.csv", "FB_SVM_DT_None.csv", "FB_NN_DT_None.csv"],
        ["FB_RF_DT_SNV.csv", "FB_SVM_DT_SNV.csv", "FB_NN_DT_SNV.csv"],
        ["FB_RF_DT_MSC.csv", "FB_SVM_DT_MSC.csv", "FB_NN_DT_MSC.csv"],
        ["FB_RF_DT_SGFD.csv", "FB_SVM_DT_SGFD.csv", "FB_NN_DT_SGFD.csv"]]

r2 = [['0.7807', '0.7473', '0.7789'],
      ['0.8430', '0.7917', '0.8117'],
      ['0.8617', '0.7850', '0.8678'],
      ['0.8872', '0.8207', '0.8638']]

mae = [['0.3203', '0.3713', '0.3424'],
      ['0.3006', '0.3353', '0.2987'],
      ['0.2709', '0.3190', '0.2424'],
      ['0.2483', '0.2996', '0.2803']]

rmse = [['0.4491', '0.4821', '0.4509'],
      ['0.3799', '0.4377', '0.4161'],
      ['0.3566', '0.4447', '0.3487'],
      ['0.3220', '0.4060', '0.3539']]
'''

figsize = 30,35
plt.figure(figsize=figsize)

for i in range(4):
    for j in range(3):
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/全波段/"+ file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/全波段纹理/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/PCA/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/回归系数/" + file[i][j], encoding='utf-8')
        df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/SPA/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/PCA纹理/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/回归系数纹理/" + file[i][j], encoding='utf-8')
        #df = pd.read_csv(r"/Users/shaoqi/Desktop/paper/结果/多糖/SPA纹理/" + file[i][j], encoding='utf-8')


        ax = plt.subplot(4, 3, index[i][j])
        
        y1 = df['DT'].to_list()
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
        
        p0, = plt.plot([3,8], [3,8], '--', color='black', label='',linewidth = 1.0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='$\mathregular{R^2}$ = '+r2[i][j],linewidth = 0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='RMSE = ' + rmse[i][j],linewidth = 0)
        p0, = plt.plot([3,8], [3,8], '', color='black', label='MAE = ' + mae[i][j],linewidth = 0)
        
        ############# 设置坐标刻度值的大小以及刻度值的字体 #############
        plt.xlim(3, 8)
        plt.ylim(3, 8)
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


plt.savefig('SPA_DT.eps', dpi=2000)
plt.show()
