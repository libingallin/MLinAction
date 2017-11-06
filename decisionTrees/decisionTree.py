# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:15:08 2017

@author: libing

decision trees:
    data set divided by information gain(IDE3)
"""
import math
import operator


def createDataSet():
    '''《统计学习方法》上的例子'''
    dataSet = [['young', 'no', 'no', 'normal', 0],
               ['young', 'no', 'no', 'good', 0],
               ['young', 'yes', 'no', 'good', 1],
               ['young', 'yes', 'yes', 'normal', 1],
               ['young', 'no', 'no', 'normal', 0],
               ['middleaged', 'no', 'no', 'normal', 0],
               ['middleaged', 'no', 'no', 'good', 0],
               ['middleaged', 'yes', 'yes', 'good', 1],
               ['middleaged', 'no', 'yes', 'great', 1],
               ['middleaged', 'no', 'yes', 'great', 1],
               ['elder', 'no', 'yes', 'great', 1],
               ['elder', 'no', 'yes', 'good', 1],
               ['elder', 'yes', 'no', 'good', 1],
               ['elder', 'yes', 'no', 'great', 1],
               ['elder', 'no', 'no', 'normal', 0]]
    labels = ['age', 'work', 'house', 'creidt']
    return dataSet, labels


"""def createDataSet():
    '''<ML in Action> example'''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['x', 'f']
    return dataSet, labels
"""


def calcShannonEnt(dataSet):
    '''计算给定数据集的熵'''
    numEntries = len(dataSet)  # 数据集中实例总数
    labelCounts = {}  # (类别：出现次数)
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt  # 数据集的熵


def splitDataSet(dataSet, axis, value):
    '''按照给定特征(数据第axis个特征取值value)划分数据集'''
    retDataSet = []
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeat2Split(dataSet):
    '''选择最好的数据集划分方式
        信息增益(gain information)
    '''
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)  # 数据集的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历每一个特征
        uniqueVals = set([example[i] for example in dataSet])  # 第i个特征的所有取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # print infoGain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 类别标签列表
    if classList.count(classList[0]) == len(classList):  # 数据集类别完全相同
        return classList[0]
    if len(dataSet[0]) == 1:   # 遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeat2Split(dataSet)  # 选择最佳划分特征
    bestFeatLabel = labels[bestFeat]  # 对应的类别标签
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 遍历当前选择特征包含的所有属性值
        # 函数参数是列表时，参数是按照引用方式传递的
        # 为了确保每次调用函数createTree()时不改变原始列表的内容
        # 使用新变量subLabels代替原始列表
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
                splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归
    return myTree


if __name__ == '__main__':
    mydata, labels = createDataSet()
    myTree = createTree(mydata, labels)
    print myTree






















