# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:15:08 2017

@author: libing

decision trees: 
    data set divided by information gain
    
"""
import operator
from math import log


def calcShannoEnt(dataSet):
    '''calculate ShannoEnt(information entropy) of data set'''
    numEntries = len(dataSet) #the number of dataSet
    labelCounts = {} #categroies dict
    for featVec in dataSet:  #find all kinds
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts: #calculate shannoEnt
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt

def splitDataSet(dataSet, axis, value):
    '''split data set'''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:]) 
            retDataSet.append(reduceFeatVec) #every element is a list 
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    '''information gain'''
    numEntries = len(dataSet[0]) - 1 #数据的最后一个是类别，不予考虑
    baseEntropy = calcShannoEnt(dataSet) #information entropy of dataSet
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numEntries): 
        featList = [example[i] for example in dataSet] #每一个属性的所有取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''vote to decide class:the minority is subordinate to the majority'''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] + 1
    sortedClassCounts = sorted(classCount.iteritems(), \
                               key=operator.itemgetter(1), reverse=True)
    return sortedClassCounts[0][0]     
           
def createTree(dataSet, labels):
    '''create decision tree'''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]         #classes of all datas are same,then stop
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues =  [example[bestFeat] for examplt in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, \
              bestFeat, value), subLabels) 
    return myTree
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


