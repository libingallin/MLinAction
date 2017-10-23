# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:52:12 2017

@author: libing
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    '''逐行读取数据，每行的前两个值分别是X1和X2，第三个值是
        数据对应的类别标签。为方便计算，该函数还将X0的值设为1.0'''
    dataMat = []  # 数据集
    labelMat = []  # 数据标签
    fr = open('testSet.txt')
    for line in fr.readlines():  # 逐行读取数据
        # strip返回去除两侧空白的字符串
        # split不提供任何分隔符，把所有空白作为分隔符(制表符、空白、换行等)
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # list
        labelMat.append(int(lineArr[2]))  # list
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
    for k in range(maxCycles):  # 每次更新权值都需要遍历整个数据集
        h = sigmoid(dataMatrix * weights)  # 输出
        error = labelMat - h  # 训练误差
        weights = weights + alpha * dataMatrix.transpose() * error  # 更新权值
    return weights


def plotBestFit(weights):
    '''画出数据集和Logistic回归最佳拟合直线的函数'''
    dataMat, labelMat = loadDataSet()  # 数据和标签
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]  # 数据个数
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:  # 类别为1
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:  # 类别为2
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]  # boundary decision
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels, alpha=0.01):
    ''' Stochatic Gradient Ascent Algorithm
        learning rate: alpha'''
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for i in range(m):  # 每次仅用一个样本点来更新回归系数(权值)
        h = sigmoid(np.sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''Modified Stochatic Gradient Ascent Algorithm'''
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):  # 迭代次数
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01  # alpha每次迭代的时候都会调整
            # 随机选取样本来更新回归系数
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
            del dataIndex[randIndex]
    return weights


# 用logistic regression进行分类
def classifyVector(inX, weights):
    '''以回归系数和特征向量作为输入来计算对应的Sigmoid值。
        如果大于0.5，则函数返回1，否则返回0'''
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1
    return 0


def colicTest():
    '''打开测试集和训练集，并对数据进行格式化处理'''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 训练集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 测试集
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != \
           int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print 'the error rate of this test is: %f' % errorRate
    return errorRate


def multiTest():
    '''调用colicTest() 10次并求结果的平均值'''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % \
        (numTests, errorSum/float(numTests))
