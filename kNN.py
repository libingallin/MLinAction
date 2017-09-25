    # -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 09:39:23 2017

@author: libing

kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

"""

import numpy as np
import os, operator


def createDataSet():
    '''
        create a sample data set
    '''
    group = np.array(([1., 1.1], [1., 1.], [0., 0.], [0., 0.1]))
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify(inX, dataSet, labels, k):
    '''parameters:
        inX: vector to compare to existing dataset (1xN)
        dataSet: size m data set of known vectors (NxM)
        labels: data set labels (1xM vector)
        k: number of neighbors to use for comparison (should be an odd number)
    '''
    #计算待判别点和数据集中的每个点的距离(欧氏距离)
    dataSetSize = dataSet.shape[0] #数据集的规模（个数）
    #把inX扩展成和dataSet一样的shape, 再与dataSet相减（元素级）
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #返回 能给distance由小到大排序 的索引（去看np.argsort())
    sortedDistIndices = distances.argsort()
    classCount = {}
    #选择最小距离的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.iteritems(), 
                    key=operator.itemgetter(1), reverse=True)
    #少数服从多数
    return sortedClassCount[0][0]


def file2array(filename):
    '''process data：
            read data fram text file
        input: filename 
        output: training data set and labels
    '''
    fr = open(filename)
    arrayOLines = fr.readlines() 
    numberOfLines = len(arrayOLines) #filename里面内容的行数
    returnedArray = np.zeros((numberOfLines, 3)) #给定空训练集数组
    classLabelVector = [] #与训练集对应的标签
    index = 0
    for line in arrayOLines:
        line = line.strip() #去掉两侧的空格，返回字符串
        listFromLine = line.split('\t') #用\t将字符串分割成列表
        returnedArray[index, :] = listFromLine[0: 3] #初始化训练集数组
        classLabelVector.append(int(listFromLine[-1])) # 初始化labels
        #如果不用int，将会当成字符串处理
        index += 1
    return returnedArray, classLabelVector


def autoNorm(dataSet):
    '''
    data normalization
    newvalue = (oldvalue-min) / (max-min)
    '''
    minValues = dataSet.min(0) #每个特征的最小值
    maxValues = dataSet.max(0) #每个特征的最大值
    ranges = maxValues - minValues 
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minValues    
    
    
def datingClassTest():
    hoRation = 0.10
    #从文本文件中解析数据
    datingDataArr, datingLabels = file2array('datingTestSet2.txt')
    #归一化特征值
    normArr, ranges, minValues = autoNorm(datingDataArr)
    m = normArr.shape[0]
    numTestVecs = int(m*hoRation) #测试数据个数
    errorCount = 0.0 #测试时分类错误数据个数
    #测试
    for i in range(numTestVecs):
        classifierResult = classify(normArr[i, :], 
                                    normArr[numTestVecs:m,:],
                                    datingLabels[numTestVecs:m], 4)
        print "the classifier came back with: %d, the real answer is %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print 'the total error rate is: %f' % \
                                (errorCount/float(numTestVecs))
    
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(
                'Percentage of time spent playing video games?'))
    ffMiles = float(raw_input(
                'frequent flier miles earned per year?'))
    iceCream = float(raw_input(
                'liters of ice cream consumed per year?'))
    datingDataArr, datingLabels = file2array('datingTestSet2.txt')
    normArr, ranges, minValues = autoNorm(datingDataArr)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minValues)/ranges, 
                                   normArr, datingLabels, 3)
    print "You will probably like this person: ", \
                resultList[classifierResult - 1]
    
    
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
    
   
def handwritingClassTest():
    hwLabels = []
    #load the training set
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingArr = np.zeros((m, 1024))
    for i in range(1024):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  #read the txt name
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingArr[i, :] = img2vector(
                                'trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifyResult = classify(vectorUnderTest, trainingArr, 
                                  hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d"   % (classifyResult, classNumStr)
        if classifyResult != classNumStr:
            errorCount += 1
    print "\nthe total number of errors is: %d" % errorCount
    print '\nthe total error rate is: %f' %  (errorCount/mTest)
    