import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {'family': 'Times New Roman',
        'size': 25,
        }
sns.set(font_scale=2.0)
plt.rc('font',family='Times New Roman')

x1 = [[92.52, 89.38, 90.93],
      [87.72, 86.90, 87.31],
      [91.89, 89.20, 90.53],
      [87.18, 88.32, 87.74],
      [86.81, 90.01, 88.38],
      [84.64, 87.89, 86.23]]

x2 = [[93.53, 93.04, 93.28],
      [90.71, 89.15, 89.93],
      [93.76, 89.54, 91.60],
     [89.19, 89.51, 89.35],

     [89.25, 91.97, 90.59],
     [84.79, 88.18, 86.54]]

x3 = [[93.89, 93.56, 93.72],
      [91.03, 89.95, 90.49],
[93.74, 90.89, 92.29],
     [91.61, 89.75, 90.67],

     [90.98, 93.05, 92.00],
     [86.18, 90.60, 88.33]]

x4 = [[93.32, 94.87, 94.09],
      [92.48, 91.14, 91.81],
[94.69, 92.35, 93.51],
     [91.71, 89.63, 90.66],

     [91.99, 93.49, 92.73],
     [88.75, 91.03, 89.87]]

x5 = [[94.83, 95.74, 95.28],
      [93.83, 92.46, 93.14],
[95.62, 93.36, 94.48],
      [93.21, 91.66, 92.43],

     [93.03, 94.25, 93.64],
     [90.21, 93.16, 91.66]]

x6 = [[95.93, 96.43, 96.18],
      [95.16, 93.65, 94.40],
[96.90, 94.83, 95.85],
     [94.90, 93.21, 94.05],

     [94.04, 95.98, 95.00],
     [91.98, 94.73, 93.33]]

x=[x1, x2, x3, x4, x5, x6]
figsize = 32, 22
plt.figure(figsize=figsize)

for i in range(1, 7):

    ax = plt.subplot(3, 2, i)

    ax.xaxis.tick_top()
    h=sns.heatmap(x[i-1], annot=True, cmap='jet', linewidths=1.5, vmin=84, vmax=98, fmt='.2f', cbar=False,
                annot_kws={'family': 'Times New Roman', 'size':24, 'weight':'bold', 'color':'black'})

    cbar = h.figure.colorbar(h.collections[0])
    cbar.set_ticks(np.linspace(84, 98, 8))
    cbar.set_ticklabels(('84.0', '86.0', '88.0', '90.0', '92.0',
                         '94.0', '96.0', '98.0'))


    label=['ResNet', 'ResNeXt', 'SE-ResNet', 'SE-ResNeXt', 'SK-ResNet', 'SK-ResNeXt']
    ax.set_yticklabels(label, fontsize=28)
    labels = ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.yticks(rotation=360)


    label=['Precision', 'Recall', 'F1-score']
    ax.set_xticklabels(label, fontsize=28)
    labels = ax.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

plt.savefig('all_result.eps', dpi=1000)
plt.show()
