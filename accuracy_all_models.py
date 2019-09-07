# -*- coding: utf-8 -*-

####################### load packages ########################
import numpy as np
import matplotlib.pyplot as plt


####################### data ########################
width = 0.6
width1 = 0.85

accuracy_x1 = np.arange(0, width1*5, width1)
print(accuracy_x1)


accuracy_x2 = np.arange(0, width1*4, width1)
print(accuracy_x2)


accuracy_fs = (78.33, 93.33, 74.44, 75.00, 73.89)

accuracy_fst = (84.44, 80.56, 78.89, 76.11)

accuracy_es = (81.11, 94.44, 77.78, 77.22, 75.56)

accuracy_est = (86.67, 81.67, 79.44, 76.67)

####################### 设置输出的图片大小 #######################
figsize = 20, 12
figure, ax = plt.subplots(figsize=figsize)


############# 绘制直方图 #############
p1 = plt.bar(accuracy_x1, accuracy_fs, width, color=['mediumslateblue', 'dodgerblue', 'limegreen', 'greenyellow', 'gold'], edgecolor=None)

p2 = plt.bar(accuracy_x2+width1*5.5, accuracy_fst, width, color=['mediumslateblue', 'limegreen', 'greenyellow', 'gold'], edgecolor=None)

p3 = plt.bar(accuracy_x1+width1*10, accuracy_es, width, color=['mediumslateblue', 'dodgerblue', 'limegreen', 'greenyellow', 'gold'], edgecolor=None)

p4 = plt.bar(accuracy_x2+width1*15.5, accuracy_est, width, color=['mediumslateblue', 'limegreen', 'greenyellow', 'gold'], edgecolor=None)


############# 设置图例并且设置图例的字体及大小 #############
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 25,
         }

plt.legend((p1), ('CF', 'MGSCF', 'GBDT', 'RF','SVM'),
           ncol=1, loc=1, prop=font1, frameon=False)  # 图例

############# 设置坐标刻度值的大小以及刻度值的字体 #############
plt.xlim(-0.5, 16.2)
#plt.xticks([])

x = []
x.extend(accuracy_x1)
x.extend(accuracy_x2+width1*5.5)
x.extend(accuracy_x1+width1*10)
x.extend(accuracy_x2+width1*15.5)


y = []
y.extend(['CF-FS', 'MGSCF-FS', 'GBDT-FS', 'RF-FS', 'SVM-FS'])
y.extend(['CF-FST', 'GBDT-FST', 'RF-FST', 'SVM-FST'])
y.extend(['CF-ES', 'MGSCF-ES', 'GBDT-ES', 'RF-ES', 'SVM-ES'])
y.extend(['CF-EST', 'GBDT-EST', 'RF-EST', 'SVM-EST'])


plt.xticks(x, y, rotation=50)
plt.tick_params(labelsize=16)
labels = ax.get_xticklabels()
[label.set_fontname('Times New Roman') for label in labels]


plt.ylim(70, 100)
plt.ylabel('Accuracy(%)', font1)
plt.tick_params(labelsize=20)

labels = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

############# 直方图数据显示 #############
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }


for a, b in zip(accuracy_x1, accuracy_fs):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')

for a, b in zip(accuracy_x2+width1*5.5, accuracy_fst):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')

for a, b in zip(accuracy_x1+width1*10, accuracy_es):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')

for a, b in zip(accuracy_x2+width1*15.5, accuracy_est):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')


############# 去掉边框 #############
#plt.gca().spines['top'].set_visible(False) # 去掉上边框
#plt.gca().spines['right'].set_visible(False) # 去掉右边框

############# 保存输出 #############
plt.savefig('C:\\Users\\shaoqi\\Desktop\\classification.eps', dpi=400)

plt.show()