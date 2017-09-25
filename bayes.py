# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:47:26 2017

@author: libing

naive bayes classifier

"""

import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1 points to bad, 0 points to normal
    return postingList, classVec

def createVocabList(dataSet):
    '''create vocabulary via dataSet'''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''input: vocabulary and a input text
       output: vector of input text, element is 0 or 1,
           1 point to appearance, 0 not'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print ('the word: %s is not in  my vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''naive bayes train'''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        










    