python画图相关tips
================

# 设置字体为times new roman

> 代码及结果如下：

```python
# -*- coding: utf-8 -*-
import  matplotlib.pyplot as plt  

#数据设置  
x1 =[0,5000,10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000];  
y1=[0, 223, 488, 673, 870, 1027, 1193, 1407, 1609, 1791, 2113, 2388];  
  
x2 =[0,5000,10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000];  
y2=[0, 214, 445, 627, 800, 956, 1090, 1281, 1489, 1625, 1896, 2151];  
  
#设置输出的图片大小  
figsize = 11,9  
figure, ax = plt.subplots(figsize=figsize)  
  
#在同一幅图片上画两条折线  
A,=plt.plot(x1,y1,'-r',label='A',linewidth=5.0)  
B,=plt.plot(x2,y2,'b-.',label='B',linewidth=5.0)  
  
#设置图例并且设置图例的字体及大小  
font1 = {'family' : 'Times New Roman',  
'weight' : 'normal',  
'size'   : 23,  
}  
legend = plt.legend(handles=[A,B],prop=font1)  
  
#设置坐标刻度值的大小以及刻度值的字体  
plt.tick_params(labelsize=23)  
labels = ax.get_xticklabels() + ax.get_yticklabels()  
[label.set_fontname('Times New Roman') for label in labels]  
  
#设置横纵坐标的名称以及对应字体格式  
font2 = {'family' : 'Times New Roman',  
'weight' : 'normal',  
'size'   : 30,  
}  
plt.xlabel('round',font2)  
plt.ylabel('value',font2)  

plt.title('A and B',fontdict=font2)

plt.show()
```

![image](https://github.com/ShaoQiBNU/python_plot_tips/blob/master/image/1.tiff)

# 直方图及样式填充

> 代码及结果如下：

```python
# -*- coding: utf-8 -*-

####################### load packages ########################
import numpy as np
import matplotlib.pyplot as plt


####################### data ########################
N = 3
width = 0.5
wid = 0.4

accuracy_x = np.arange(0, N*4, 4)
recall_x = np.arange(0, N*4, 4) + wid*3
F1_x = np.arange(0, N*4, 4) + wid*6


accuracy_rf = (0.71, 0.88, 0.64)
accuracy_plsda = (0.66, 0.85, 0.64)

recall_rf = (0.67, 0.95, 0.63)
recall_plsda = (0.70, 1.00, 0.48)

F1_rf = (0.69, 0.91, 0.64)
F1_plsda = (0.68, 0.92, 0.55)


####################### 设置输出的图片大小 #######################
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)

############# 直方图填充样式 #############
patterns = ('//', '||', '\\\\')

############# 绘制直方图-为了图例的输出 #############
p01 = plt.bar(accuracy_x, accuracy_rf, width, color='#9999ff', edgecolor='black')
p02 = plt.bar(accuracy_x + width, accuracy_plsda, width, color='violet', edgecolor='black')

p03 = plt.bar(accuracy_x, accuracy_rf, width, color='white', edgecolor='black', hatch=patterns[0])
p04 = plt.bar(accuracy_x + width, accuracy_plsda, width, color='white', edgecolor='black', hatch=patterns[1])
p05 = plt.bar(accuracy_x + width, accuracy_plsda, width, color='white', edgecolor='black', hatch=patterns[2])

############# 绘制直方图 #############
p1 = plt.bar(accuracy_x, accuracy_rf, width, color='#9999ff', edgecolor='black', hatch=patterns[0])
p2 = plt.bar(accuracy_x + width, accuracy_plsda, width, color='violet', edgecolor='black', hatch=patterns[0])

p3 = plt.bar(recall_x, recall_rf, width, color='#9999ff', edgecolor='black', hatch=patterns[1])
p4 = plt.bar(recall_x + width, recall_plsda, width, color='violet', edgecolor='black', hatch=patterns[1])

p5 = plt.bar(F1_x, F1_rf, width, color='#9999ff', edgecolor='black', hatch=patterns[2])
p6 = plt.bar(F1_x + width, F1_plsda, width, color='violet', edgecolor='black', hatch=patterns[2])


############# 设置图例并且设置图例的字体及大小 #############
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }

plt.legend((p01[0], p02[0], p03[0], p04[0], p05[0]), ('', '', '', '', ''),
           loc=2, prop=font1, frameon=False)  # 图例

############# 设置坐标刻度值的大小以及刻度值的字体 #############
plt.xlim(-0.5, 11.4)
plt.xticks([])

plt.ylim(0.4, 1)

plt.tick_params(labelsize=18)

labels = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

############# 直方图数据显示 #############
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }

for a1, a2, b, c in zip(accuracy_x, accuracy_x + width, accuracy_rf, accuracy_plsda):
    plt.text(a1, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')
    plt.text(a2, c, '%.2f' % c, ha='center', va='bottom', fontdict=font2, color='black')

for a1, a2, b, c in zip(recall_x, recall_x + width, recall_rf, recall_plsda):
    plt.text(a1, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')
    plt.text(a2, c, '%.2f' % c, ha='center', va='bottom', fontdict=font2, color='black')

for a1, a2, b, c in zip(F1_x, F1_x + width, F1_rf, F1_plsda):
    plt.text(a1, b, '%.2f' % b, ha='center', va='bottom', fontdict=font2, color='black')
    plt.text(a2, c, '%.2f' % c, ha='center', va='bottom', fontdict=font2, color='black')


############# 去掉边框 #############
plt.gca().spines['top'].set_visible(False) # 去掉上边框
plt.gca().spines['right'].set_visible(False) # 去掉右边框

############# 保存输出 #############
plt.savefig('/Users/shaoqi/Desktop/1.tiff', dpi=400)

plt.show()
```

![image](https://github.com/ShaoQiBNU/python_plot_tips/blob/master/image/2.tiff)
