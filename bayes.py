# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:47:26 2017

@author: libing

naive bayes classifier

"""
from math import log
import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park',
                    'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop',
                    'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 points to bad, 0 points to normal
    return postingList, classVec


def createVocabList(dataSet):
    '''create vocabulary via dataSet'''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''input:  vocabulary and a input text
       output: vector of input text, element is 0 or 1,
                1 point to appearance, 0 not'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print ('the word: %s is not in  my vocabulary!' % word)
    return returnVec


def trainNB0(trainArr, trainCategory):
    ''' 朴素贝叶斯训练函数
        输入参数为文档矩阵trainArr,以及
        每篇文档类别标签构成的向量trainCategory'''
    numTrainDocs = len(trainArr)  # 文档矩阵中的文档数目
    numWords = len(trainArr[0])   # 每篇文档的特征个数
    # 计算先验概率P(c)，二分类问题，所以只需计算一个类别的，另一个用1减去这个
    pAbusie = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 类别为1
            p1Num += trainArr[i]  # 统计文档中出现的各个单词个数
            p1Denom += np.sum(trainArr[i])  # 统计文档的总单词个数
        else:  # 类别为0
            p0Num += trainArr[i]
            p0Denom += np.sum(trainArr[i])
    p1Vect = log(p1Num/p1Denom)  # 类别1中各单词频率
    p0Vect = log(p0Num/p1Denom)  # 类别0中各单词频率
    return p0Vect, p1Vect, pAbusie


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''朴素贝叶斯分类函数'''
    p1 = np.sum(vec2Classify * p1Vec) + log(pClass1)  # 各特征概率
    p0 = np.sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  # 各特征概率
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 测试文档及其类别向量
    myVocabList = createVocabList(listOPosts)  # 词汇表
    trainArr = []
    for postinDoc in listOPosts():
        trainArr.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainArr), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'Classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'Classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)



























