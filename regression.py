# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:22:28 2017

@author: libing
"""
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2


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
        这种方法称为OLS，即普通最小二乘法(ordinary least squares)
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 计算行列式，det(xTx)==0,不可逆
        print("This matrix is singular, cannot do inverse!")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws  # 回归系数


def testing():
    '''这是一个便利函数(convenience function),该函数封装所有操作，以节省时间。
        线性回归(OLS)的测试
    '''
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
        LWLR增加了计算量，对每个点预测时都必须使用整个数据集
    '''
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
    '''LWLR算法的测试'''
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


def rssError(yArr, yHatArr):
    '''分析预测误差的大小'''
    return ((yArr-yHatArr)**2).sum()


def testing3():
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[:99], abX[0:99], abY[0:99], 0.1)  # k=0.1
    error01 = rssError(abY[0:99], yHat01.T)
    yHat1 = lwlrTest(abX[:99], abX[0:99], abY[0:99], 1)  # k=1
    error1 = rssError(abY[0:99], yHat1.T)
    yHat10 = lwlrTest(abX[:99], abX[0:99], abY[0:99], 10)  # k=10
    error10 = rssError(abY[0:99], yHat10.T)
    print error01, error1, error10
    # 在新数据上的比较
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)  # k=0.1
    error01 = rssError(abY[100:199], yHat01.T)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)  # k=1
    error1 = rssError(abY[100:199], yHat1.T)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)  # k=10
    error10 = rssError(abY[100:199], yHat10.T)
    print error01, error1, error10
    # 与简单的线性回归作比较
    ws = standRegess(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print rssError(abY[100:199], yHat.T.A)


def ridgeRegres(xMat, yMat, lam=0.2):
    '''岭回归(ridge regression).解决数据的特征数比样本点还多。计算回归系数'''
    xTx = xMat.T*xMat
    denom = xTx + np.eye(xMat.shape[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)  # weights
    return ws


def ridgeTest(xArr, yArr):
    '''给定lambda下的岭回归求解
        数据标准化：
        zero-mean normalization: (x-mean)/std
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)  # mean
    xVar = np.var(xMat, 0)  # variance
    xMat = (xMat-xMeans) / xVar  # 数据标准化
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    '''regularize by columns'''
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)
    inVar = np.var(xMat, 0)
    return (inMat-inMeans) / inVar


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''前向逐步回归：贪心算法，每一步都尽可能减少误差。开始时，所有权重都设为1，然后
        每一步所做的决策是对某个权重增加或减少一个很小的值。
        eps：每次迭代需要调整的步长
        numIt：迭代次数
    '''
    xMat = np.mat(xArr)  # (4177, 8)
    yMat = np.mat(yArr).T  # (1, 4177)
    # 数据标准化
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = xMat.shape  # m: 样本数， n：特征数
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):  # 每次迭代
        print ws
        lowestError = np.inf
        for j in range(n):  # 对于每个特征
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign  # 改变系数得到新的w
                yTest = xMat * wsTest  # 计算预测值
                rssE = rssError(yMat.A, yTest.A)  # 计算误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 代码跑不出来，以下代码看懂了，但没有跑。
# mark一下。等学了urllib2再来跑。
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)  # 休眠10s，是为了防止短时间内有过多的API调用
    myAPIstr = 'get from code.google.com'
    searchURL = 'http://www.googleapis.com/shopping/search/v1/public/products?\
                key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['item'])):
        try:
            currItem = retDict['item'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc*0.5:  # 过滤掉不完整的套装
                    print("%d\t%d\t%d\t%f\t%f" %
                          (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    '''交叉验证测试岭回归'''
    m = len(xArr)  # 总样本个数
    indexList = range(m)
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):  # 交叉验证
        trainX = []
        trainY = []
        testX = []
        testY = []
        # Modify a sequence in-place by shuffling its contents
        np.random.shuffle(indexList)  # 混洗、打乱，实现随机选取
        for j in range(m):
            if j < 0.9*m:
                # train set, 90%
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                # test set, 10%
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # 岭回归求得回归系数
        for k in range(30):  # 30个不同的lambda值创建了20组不同的回归系数
            # 测试数据标准化
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTrainX-meanTrain) / varTrain
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # 预测
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))  # 误差
    # 为了将得出的回归系数与standRegrs()作比较,需要计算这些误差估计的均值
    # 此处为10折，所以每个lambda对应10个误差。应选取使误差的均值最低的回归系数
    meanErrors = np.mean(errorMat, 0)
    minMean = float(np.min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # 测试数据标准化
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    # 岭回归使用了数据标准化，而standRegres()没有
    # 因此为了将上述比较可视化还需将数据还原
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ",
          -1*np.sum(np.multiply(meanX, unReg)), np.mean(yMat))
