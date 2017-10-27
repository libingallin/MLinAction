# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:22:28 2017

@author: libing
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    '''读取数据，每行最后一个值为目标值'''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        lineArr = []
        curLine = line.rstrip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))   # 目标值
    return dataMat, labelMat


def standRegess(xArr, yArr):
    '''计算最佳拟合直线。采用平方误差。
        w = (X.T*X).I * (X.T*y)
        这种方法称为OLS，即普通最小二乘法(ordinary least squares)'''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 计算行列式，det(xTx)==0,不可逆
        print("This matrix is singular, cannot do inverse!")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws  # 回归系数


def testing():
    '''这是一个便利函数(convenience function),该函数封装所有操作，以节省时间'''
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegess(xArr, yArr)  # 回归系数
    # 数据集中的数据第一个值全是1(X0=1)，则y=ws[0]+ws[1]*X1
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws  # 预测值
    # 绘制数据集散点图和最佳拟合直线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    # 直线上的数据点次序混乱，绘图时将会出现问题
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, c='r')


# 线性回归的一个问题是有可能出现欠拟合现象，因为它求的是具有最小均方误差的无偏估计。
# 故有些方法允许在估计中引入一些偏差，从而降低预测的均方误差。
# 其中一个方法是局部加权线性回归(Locially Weighted Linear Regression,LWLR)
def lwlr(testPoint, xArr, yArr, k=1.0):
    '''局部加权线性回归(Locially Weighted Linear Regression,LWLR)
        给待预测点附近的每个点赋予一定的权值，在这个自己上基于最小均方差来进行普通的
        回归。算法需要事先选取出对应的数据子集。解出回归系数：
            w = (X.T*W*X).I * (X.T*W*y)
        使用核来对附近的点赋予更高的权重。核的类型自由选择，常用的是高斯核，对应权重：
        w(i,i) = exp(|x(i)-x|/(-2*K**2))
        用户指定的k值，决定了对附近的点赋予多大的权重，这是使用LWLR时唯一考虑的参数
        LWLR增加了计算量，对每个点预测时都必须使用整个数据集'''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]  # 样本点个数
    weights = np.mat(np.eye(m))  # 对角矩阵，初始化一个权重
    # 遍历数据集，计算每个样本点对应的权值：随着样本点与待预测点距离的递增，
    # 权重将以指数级衰减。参数k控制衰减的速度。
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # 高斯核对应的权重
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is sigular, cannot do inverse!")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def testing2():
    xArr, yArr = loadDataSet('ex0.txt')
    # 对单点进行估计
    output0 = lwlr(xArr[0], xArr, yArr, 0.001)
    print(output0)
    # 得到数据集里所有点的估计
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    # return the indices that would sort the matrix, axis=0
    strInd = xMat[:, 1].argsort(0)
    xSort = xMat[strInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[strInd], c='r')
    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2)


















    
