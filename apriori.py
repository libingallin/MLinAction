# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:16:17 2017

@author: libing
"""


def loadDataSet():
    '''create a simple test set'''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []  # 存储所有不重复的项指
    for transction in dataSet:  # 遍历数据集中每项交易记录
        for item in transction:  # 遍历记录中的每一个项
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # print C1
    return map(frozenset, C1)  # 冻结C1中的每一项，即不可改变


def scanD(D, Ck, minSupport):
    '''D: 数据集
       Ck: 候选集列表
       minSupport: 感兴趣项集的最小支持度
    '''
    ssCnt = {}  # 各项的支持度
    for tid in D:  # 数据集中的每一项记录
        for can in Ck:  # 候选集列表中的每一项(项集)
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    '''创建候选集列表Ck
        Lk: 频繁项集列表
        k: 项集元素个数'''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # ‘|’ 集合的并集
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
