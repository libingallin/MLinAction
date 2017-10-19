# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:52:12 2017

@author: libing
"""

import numpy as np


def loadDataSet(dataSePath):
    '''逐行读取数据(路径为dataSetPath)，每行的前两个值分别是X1和X2，第三个值是
        数据对应的类别标签。为方便计算，该函数还将X0的值设为1.0'''
    dataMat = []  # 数据集
    labelMat = []  # 数据标签
    fr = open(dataSePath)
    for line in fr.readlines():  # 逐行读取数据
        # strip返回去除两侧空白的字符串
        # split不提供任何分隔符，把所有空白作为分隔符(制表符、空白、换行等)
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    '''Sigmoid function'''
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''Gradient Ascent Algorithm
        dataMatIn: 二维数组，每列代表不同的特征，每行代表每个训练样本'''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 步长
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 输出
        error = labelMat - h  # 训练误差
        weights = weights + alpha * dataMatrix.transpose() * error  # 更新权值
    return weights
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    