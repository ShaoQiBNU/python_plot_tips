import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

font = {'family': 'Times New Roman',
        'size': 25,
        }
sns.set(font_scale=2.0)
plt.rc('font',family='Times New Roman')

x1 = [[88.46, 88.62, 88.52, 86.41, 88.73],
     [90.21, 90.23, 90.20, 88.52, 90.48],
     [91.24, 91.30, 91.25, 89.74, 91.49],
     [92.16, 92.08, 92.11, 90.72, 92.31],
     [93.45, 93.44, 93.44, 92.28, 93.60],
     [94.82, 94.80, 94.80, 93.89, 94.93]]

x2 = [[83.09, 83.19, 83.12, 80.01, 83.42],
     [86.96, 87.06, 86.98, 84.64, 87.27],
     [87.95, 87.97, 87.92, 85.81, 88.24],
     [90.40, 90.41, 90.40, 88.73, 90.65],
     [91.31, 91.33, 91.29, 89.78, 91.53],
     [92.73, 92.62, 92.66, 91.41, 92.88]]

x3 = [[82.68, 82.84, 82.72, 79.61, 83.08],
     [86.58, 86.62, 86.56, 84.16, 86.87],
     [86.84, 86.85, 86.79, 84.46, 87.11],
     [89.37, 89.25, 89.28, 87.46, 89.61],
     [90.86, 90.91, 90.85, 89.19, 91.03],
     [91.64, 91.53, 91.55, 90.06, 91.76]]

x=[x1, x2, x3]
figsize = 16, 20
plt.figure(figsize=figsize)

for i in range(1, 4):

    ax = plt.subplot(3, 1, i)

    ax.xaxis.tick_top()
    h=sns.heatmap(x[i-1], annot=True, cmap='jet', linewidths=1.5, vmin=79, vmax=95, fmt='.2f', cbar=False,
                annot_kws={'family': 'Times New Roman', 'size':24, 'weight':'bold', 'color':'black'})

    cbar = h.figure.colorbar(h.collections[0])
    cbar.set_ticks(np.linspace(79, 95, 9))
    cbar.set_ticklabels(('79.0','81.0', '83.0', '85.0', '87.0', '89.0',
                         '91.0', '93.0', '95.0'))


    label=['ResNet', 'ResNeXt', 'SE-ResNet', 'SE-ResNeXt', 'SK-ResNet', 'SK-ResNeXt']
    ax.set_yticklabels(label, fontsize=28)
    labels = ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.yticks(rotation=360)


    label=['Precision', 'Recall', 'F1-score', 'Kappa', 'Accuracy']
    ax.set_xticklabels(label, fontsize=28)
    labels = ax.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

plt.savefig('noise_result.tif', dpi=100)
plt.show()
