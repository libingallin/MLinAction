# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:47:26 2017

@author: libing

Naive Bayes Classifier

"""

import operator
import feedparser
import numpy as np
import re


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park',
                    'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop',
                    'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    '''create vocabulary via dataSet'''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''该函数的输入参数为词汇表及某个文档，输出的是文档向量
       向量的每一元素为1或0，分别表示词汇表的单词在输入文档中是否出现
       每个词出现与否作为特征，这可以被描述为词集模型(set-of-words model)'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:  # 遍历文档中的所有单词
        if word in vocabList:  # 出现词汇表中的单词
            returnVec[vocabList.index(word)] = 1  # 该词是否出现
        else:
            print ('the word: %s is not in  my vocabulary!' % word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    '''词袋模型(bag-of-words model):一个词在文档中出现不止一次
        这可能意味着包含该词是否出现在文档中所不能表达的某种信息'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 该词出现次数
    return returnVec


def trainNB0(trainArr, trainCategory):
    '''朴素贝叶斯分类器训练函数
       trainArr: 经过setOfWords2Vec处理的文档
       trainCategoty: 每篇文档类别标签构成的向量
    '''
    numTrainDocs = len(trainArr)  # 文档数目
    numWords = len(trainArr[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)  # 侮辱文档(类别1)的概率
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # 为避免计算时某一个概率为0，最后的乘积也为0
    # 将所有词的出现次数初始化为1，并将分母初始化为2
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 类别为1(侮辱)的文档
            p1Num += trainArr[i]   # 出现的单词的数目
            p1Denom += np.sum(trainArr[i])  # 出现的单词总数
        else:  # 类别为0(正常)的文档
            p0Num += trainArr[i]
            p0Denom += np.sum(trainArr[i])
    # 由于太多很小的数相乘，会造成下溢出或得不到正确的答案。
    # 一种解决是对乘积取自然对数，采用自然对数进行处理不会有任何损失。
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''朴素贝叶斯分类函数
       vec2Classify: 要分类的向量
       剩下3个参数是利用trainNB0()得到的'''
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    return 0


def testingNB():
    '''这是一个便利函数(convenience function),该函数封装所有操作，以节省时间'''
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainArr = []
    for postinDoc in listOfPosts:
        trainArr.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainArr), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


def textParse(bigString):
    '''文件解析:将文档解析为字符串列表（去掉少于两个字符的字符串），
        并将所有字符串转换为小写'''
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''垃圾邮件测试函数'''
    # 导入并解析文本文件
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open(r'email\spam\%d.txt' % i).read())
        docList.append(wordList)  # 每个文档(列表)为元素组成的列表
        fullText.extend(wordList)  # 所有文档汇成一个列表
        classList.append(1)  # 类别为1的文档
        wordList = textParse(open(r'email\ham\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 类别为0的文档
    vocabList = createVocabList(docList)  # 创建词汇表
    trainingSet = range(50)  # 共50封邮件
    # 随机选取10封邮件作为测试集，同时将其从训练集剔除
    # 这种方法叫做留出法，若重复多次这种操作，即为交叉验证
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])  # 原地操作(修改trainingSet)
    # training
    trainingMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainingMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # testing
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != \
           classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float((errorCount) / len(testSet))


def calcMosttFreq(vocabList, fullText):
    '''遍历词汇表中的没个词并统计它在文本中出现的次数，然后排序（由高到低），
        最后返回次数最高的30个词'''
    freqDict = {}  # 单词频率字典
    for token in vocabList:
        freqDict[token] = fullText.count(token)  # 统计次数
        sortFreqDict = sorted(freqDict.iteritems(), key=operator.itemgetter(1),
                              reverse=True)  # 由高到低排序
    return sortFreqDict


def 














