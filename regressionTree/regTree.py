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
    return np.mat(dataMat)  # np.matrix


def binSplitDataSet(dataSet, feature, value):
    '''Split dataSet in two subclass according to given feature and value.
    Args:
        dataSet: data set, arrar_like
        feature: split
        value: the value of feature to do the split
    Returns:
        two subclasses
    '''
    # more than value
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    # less than or equal to value
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''Generate leaf node.Returns the value used for each leaf.'''
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    '''Calculate the square error of target variable on the given dataset.'''
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''Find the best feature and value to split dataSet.'''
    # parameters pointed by user
    tolS = ops[0]  # tolerant error decrease
    tolN = ops[1]  # minimun sample size
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit condition 1
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by
    # Reduction in RSS(residual sum of squares) error from mean
    S = errType(dataSet)
    bestS = np.inf  # error
    bestIndex = 0  # the best feature to split
    bestValue = 0  # the corresponding value
    for featIndex in range(n-1):  # traverse every feature
        for splitVal in set(dataSet.A[:, featIndex]):  # every value of feature
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # split
            # if the size of some subclass is less than tolN don't do the split
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
    # if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):  # I think
    # return None, leafType  # exit condition 3  # this for-loop is redundant
    # returns the best feature to split on and the value used for that split
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


def isTree(obj):
    '''Test if obj is a tree.
    Returns: boolean
    '''
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    '''Traverse tree until leafs are finded. If finded, calculate the mean.'''
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left']+tree['right']) / 2.0


def pruning(tree, testData):
    '''Prune tree.
    Args:
        tree: tree to be pruned
        testData: test data set to prune
    Returns:
        a tree after pruned or mean of tree
    '''
    if testData.shape[0] == 0:  # if we have no test data collapse the tree
        return getMean(tree)
    # if the branches are not trees try to prune them
    if (isTree(tree['left']) or isTree(tree['right'])):  # have a sub-tree
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):  # if have a left sub-tree prune it
        tree['left'] = pruning(tree['left'], lSet)
    if isTree(tree['right']):  # if have a right tree prune it
        tree['right'] = pruning(tree['right'], rSet)
    # if they are both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1]-tree['left'], 2)) + \
            np.sum(np.power(rSet[:, -1]-tree['right'], 2))
        treeMean = (tree['left']+tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1]-treeMean, 2))
        if errorMerge < errorNoMerge:  # if error decrease merge them
            print('Merging!')
            return treeMean
        else:
            return tree
    else:
        return tree


# model tree
def linearSolve(dataSet):
    '''Process data set to target variable Y and independent variable X.'''
    m, n = dataSet.shape
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]  # independent variable
    Y = dataSet[:, -1]  # target variable
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse, \n\
                        try in creasing the second value of ops")
    ws = xTx.I * (X.T * Y)  # linear regression weights
    return ws, X, Y


def modelLeaf(dataSet):
    '''Generate model of leaf when don not need to split data set.'''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    '''Calculate the square error of given data set.'''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(yHat - Y, 2))


# use tree regression to forecast
def regTreeEval(model, inData):  # regression tree
    return float(model)


def modelTreeEval(model, inData):  # model tree
    n = inData.shape[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inData
    return float(X*model)


def treeForecast(tree, inData, modelEval=regTreeEval):
    '''Forecast.
    Args:
        tree: tree
            got by createTree()
        inData: single data(a vector)
            value to forecast
        modelEval: regTreeEval, modelTreeEval, optional
            the kind of tree
    Returns:
        a scalar, forecasting tree
    '''
    if not isTree(tree):  # only root node
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForecast(tree, testData, modelEval=regTreeEval):
    '''Returns vector(forecasting value).'''
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, np.mat(testData[i]), modelEval)
    return yHat


def test():
    trainMat = loadDataSet('bikeSpeedVsIq_train.txt')  # training set
    testMat = loadDataSet('bikeSpeedVsIq_test.txt')    # testing set
    # create a regression tree
    myRegTree = createTree(trainMat, ops=(1, 20))
    yRegHat = createForecast(myRegTree, testMat[:, 0])
    print(np.corrcoef(yRegHat, testMat[:, 1], rowvar=0)[0, 1])
    # create a model tree
    myModTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yModHat = createForecast(myModTree, testMat[:, 0], modelTreeEval)
    print(np.corrcoef(yModHat, testMat[:, 1], rowvar=0)[0, 1])
    # standard line regression
    ws, X, Y = linearSolve(trainMat)
    yHat = np.mat(np.zeros((len(testMat), 1)))
    for i in range(testMat.shape[0]):
        yHat[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
