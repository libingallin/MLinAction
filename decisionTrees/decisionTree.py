# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:15:08 2017

@author: libing

decision trees:
    data set divided by information gain(IDE3)
"""
import math
import operator
import pickle


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
    '''计算给定数据集的熵(entropy)'''
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
    '''按照给定特征(数据第axis个特征的取值为value)划分数据集'''
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
            # 不同的分支节点所包含的样本数不同，给节点赋予权重prob
            # 样本数越多的分支节点的影响越大
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该子集的信息增益
        infoGain = baseEntropy - newEntropy       # gain infomation
        # infoGainRation = infoGain / baseEntropy # infomation gain ratio
        # print infoGain
        # 选择信息增益最大的特征
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


def outputTree():
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''使用决策树进行分类'''
    firstStr = inputTree.keys()[0]   # 根节点
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:  # 满足分支条件
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def testing():
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    featLabels = ['age', 'work', 'house', 'creidt']
    testVec = ['young', 'yes', 'no', 'normal']
    print classify(myTree, featLabels, testVec)
    testVec = ['young', 'no', 'no', 'great']
    print classify(myTree, featLabels, testVec)


# 将分类器存储在硬盘上，而不用每次对数据分类时重新学习一遍
# 节省时间，每次在执行分类时调用已构造好的决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


# 使用决策树预测隐形眼镜类型
def forecastLenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 嵌套列表
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree
