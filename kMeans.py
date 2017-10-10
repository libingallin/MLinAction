# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 10:15:53 2017

@author: libing
"""

import numpy as np


def loadDataSet(filename):
    '''将文本文件导入到一个列表中'''
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.rstrip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def dictEclud(vecA, vecB):
    '''Euclidean distance'''
    return np.sqrt(np.sum(np.power((vecA - vecB), 2)))


def randCent(dataSet, k):
    '''该函数为给定数据集构建一个包含k个随机质心的集合。
        随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小值
        和最大值来完成。然后生成0到0.1之间的随机数并通过取值范围和最小值，以便确保
        随机点数在数据的边界之内。
    '''
    n = np.shape(dataSet)[1]  # 每个数据的维数
    centroids = np.mat(np.zeros((k, n)))  # k个质心
    for j in range(n):
        minJ = min(dataSet[:, j])  # dataSet中第j列的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeans=dictEclud, createCent=randCent):
    m = np.shape(dataSet)[0]  # 数据集大小
    # 创建矩阵来存储每个点的簇分配结果，一列记录簇索引值，一列存储误差
    # 这里的误差是指当前点到簇质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)  # k个质心
    clusterChanged = True  # 迭代标识变量
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 对数据集中的每一个数据
            minDict = np.inf  # 无穷大
            minIndex = -1
            for j in range(k):  # 每一个质心
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDict:  # 寻找距离最近的质心
                    minDict = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 簇分配结果发生变化，更新迭代标识
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDict**2
        print centroids
        for cent in range(k):
            # 获得给定簇的所有点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算该簇下所有点的平均值，更新质心
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment
