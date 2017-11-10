# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:41:04 2017

@author: libing
"""

import matplotlib.pyplot as plt
import numpy as np


# 绘制数据集ex00.txt
def pltData():
    data = []
    fr = open('ex00.txt')
    for line in fr.readlines():
        curLine = line.rstrip().split('\t')
        fltLine = map(float, curLine)
        data.append(fltLine)
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1])
