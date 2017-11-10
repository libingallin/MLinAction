# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 19:41:38 2017

@author: libing
"""
import numpy as np


def loadDataSet(fileName):
    '''General function to parse tab-delimited floats.
        Assume laset colum is target value.
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.rstrip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return np.array(dataMat)  # np.array


def binSplitDataSet(dataSet, feature, value):
    '''Split dataSet in two subclass according to given feature and value.'''
    # more than value
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    # less than or equal to value
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''Generate leaf node.Returns the value used for each leaf.'''
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    '''Calculate the square error of target variable on the given dataset'''
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''Find the best feature and value to split dataSet.'''
    # parameters pointed by user
    tolS = ops[0]  # tolerant error decrease
    tolN = ops[1]  # minimun sample size
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].tolist())) == 1:  # exit condition 1
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by
    # Reduction in RSS(residual sum of squares) error from mean
    S = errType(dataSet)
    bestS = np.inf  # error
    bestIndex = 0  # the best feature to split
    bestValue = 0  # the corresponding value
    for featIndex in range(n-1):  # traversal every feature
        for splitVal in set(dataSet[:, featIndex]):  # every value of feature
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # split
            # if the size of some subclass is less than tolN,don't do the split
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
                continue  # jump out of this loop
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold(tolS)
    # don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit condition 2
    # split dataSet according to bestIndex and bestValue
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # if the size of some subclass is less than tolN, don't do the split
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        return None, leafType  # exit condition 3
    # returns the best feature to split on
    # and the value used for that split
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''Assume the dataSet is Numpy so we can array filtering.'''
    # choose the best split
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # if the splitting hit a stop condition return val
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)  # split
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
