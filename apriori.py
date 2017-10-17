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
    # 前k-2个项相同时，将两个集合合并
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
    '''apriori算法伪代码：
        当集合中项的个数大于0时
           构建一个k个项组成的候选项集的列表
           检查数据以确认每个项集都是频繁的
           保留频繁项集并构建K+1项组成的候选项集的列表
    '''
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)  # 字典的更新
        L.append(Lk)
        k += 1
    return L, supportData


# 频繁项集中挖掘关联规则
def generateRules(L, supportData, minConf=0.7):
    ''' L:频繁项集列表
        supportData: 包含哪些频繁项集支持数据的字典
        minConf: 最小可信度阈值
    '''
    bigRuleList = []
    # 遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
    # 无法从单元素项集中构建关联规则，所以从包含两个或更多元素的项集开始构建规则
    for i in range(1, len(L)):
        for freqSet in L[i]:  # 频繁项集列表中的每一项
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):  # 频繁项集的元素数目超过2，考虑做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 如果项集只有两个元素，计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=.7):
    '''计算规则的可信度以及找到满足最小可信度要求的规则'''
    prunedH = []
    for conseq in H:  # H中的所有候选集并计算可信度
        conf = supportData[freqSet] / supportData[freqSet-conseq]  # 计算可信度
        if conf >= minConf:  # 满足最小可信度
            print freqSet-conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH  # 返回满足最小可信度要求的规则列表


def rulesFromConseq(freqSet, H, supportData, brl, minConf=.7):
    '''freqSet: 频繁项集
        H: 可以出现在规则后件的元素列表'''
    m = len(H[0])  # 频繁集大小
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
