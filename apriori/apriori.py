# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:44:09 2017

@author: libing
"""


# generate candidate item set
def loadDataSet():
    '''Create a sample data set for testing.'''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    '''Create a list including all unique items.'''
    C1 = []
    for transction in dataSet:  # every transction
        for item in transction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()  # sort C1 inplace
    # use frozenset so we can use it as a key in a dict
    return map(frozenset, C1)  # map all elements to frozenset()


def scanD(D, Ck, minSupport):
    '''Generate a list that meets condition and a dict including every item.
    Args:
        D: datasets
        Ck: candidate set
        minSupport: minimum support
    Returns:
        a list: the support of every element is bigger than minSupport
        a dict: support of all items in Ck
    '''
    ssCnt = {}
    for tid in D:  # every data in datasets D
        for can in Ck:  # every element in candidate Ck
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 0
                ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}  # support dict
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:  # meet condition: >= minimum support
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    '''Create candidate set Ck.
    Ck is frequent item sets whose element size is k.
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=.5):
    '''Generate a list including all candidate set
    and a absolute support dict.'''
    C1 = createC1(dataSet)  # incipient candidate set
    D = map(set, dataSet)   # data set
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):  # quit when Lk is empty
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=.7):
    '''Generate rules.
    Args:
        L: list of frequent items
        supportData: a dict coming from scanD()
        minConf: minimum confidence
    Returns:
        a list of rules.
    '''
    bigRuleList = []  # rules
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:  # tranverse every frequent item
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=.7):
    # Calculate the confidence and return the rules meeting the quest(>minConf)
    pruneH = []  # create new list to return
    for conseq in H:
        # rule: freqSet-conseq --> conseq
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf > minConf:
            print freqSet-conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet-conseq, conseq, conf))
            pruneH.append(conseq)
    return pruneH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=.7):
    m = len(H[0])
    if len(freqSet) > (m+1):  # try further merging
        Hmp1 = aprioriGen(H, m+1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
