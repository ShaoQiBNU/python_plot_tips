# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:53:03 2019

@author: shaoqi
"""

import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
from matplotlib import rcParams
        
if __name__ == '__main__':
    
    # Set the figure properties (optional)
    rcParams["figure.figsize"] = [8.0, 6.4]
    rcParams['lines.linewidth'] = 1 # line width for plots
    rcParams.update({'font.size': 15, 'font.family' : 'Times New Roman'}) # font size of axes text
    
    # Close any previously open graphics windows
    # ToDo: fails to work within Eclipse
    plt.close('all')
    
    # Store statistics in arrays
    sdev = np.array([29,13,83,29,17]) # mae
    crmsd = np.array([0,62,92,55,72]) # rmse
    ccoef = np.array([1,.89,.73,.34,.83]) # r2

    # Specify labels for points in a cell array. Note that a label needs 
    # to be specified for the reference even
    label = ['Non-Dimensional Observation', 'w', 'e', 
             't', 'r']

    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel = label, 
                      locationColorBar = 'EastOutside',
                      markerDisplayed = 'colorBar', titleColorBar = 'RMSE',
                      markerLabelColor='r', markerSize=8,
                      markerLegend='off', cmapzdata=crmsd, 
                      tickSTD = range(0,120,10), axismax = 100,
                      colRMS='none', styleRMS=' ', titleRMS='off',
                      colSTD='k', styleSTD='-.', widthSTD=1.0, titleSTD ='off',
                      colCOR='b', styleCOR='--', widthCOR=1.0, titleCOR='off')
    

    # Write plot to file
    plt.savefig('C:\\Users\\shaoqi\\Desktop\\result.eps',dpi=200,facecolor='w')

    # Show plot
    plt.show()