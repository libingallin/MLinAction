# -*- coding: utf-8 -*-
"""
Use FP-growth(frequent pattern) algorithm to create FP-tree and
discover frequent item sets from FP-tree.

Created on Mon Nov 20 20:37:33 2017

@author: libing
"""


class treeNode(object):
    '''Define FP tree.'''
    def __init__(self, nameVaule, numOccur, parentNode):
        self.name = nameVaule
        self.count = numOccur
        self.nodeLink = None  # link, connect similar element
        self.parent = parentNode  # parent node, needs to be updated
        self.children = {}  # store child nodes of a node

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):  # visualize tree via text form
        print ' '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)


def loadSimpData():
    simpData = [list("rzhjp"),
                list("zyxwvuts"),
                list('z'),
                list('rxnos'),
                list('yrxzqtp'),
                list('yzxeqstm')]
    return simpData


def createInitSet(dataSet):  # dataSet formatting
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def createTree(dataSet, minSup=1):
    headerTable = {}
    # go over dataSet twice
    for trans in dataSet:  # first pass counts frequency of occurance
        for item in trans:
            # can also use another way to create headerTabel
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # print headerTable
    for k in headerTable.keys():  # remove items not meeting minSup
        if headerTable[k] < minSup:
            del headerTable[k]
    # print headerTable
    freqItemSet = set(headerTable.keys())  # frequent item sets
    if len(freqItemSet) == 0:  # if no items meet min support, get out
        return None, None
    # reformat headerTable to use Node link
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # print headerTable
    retTree = treeNode('Null Set', 1, None)  # create tree
    for tranSet, count in dataSet.items():  # go through dataset 2nd time
        localD = {}
        for item in tranSet:  # put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # print localD
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),
                            key=lambda p: p[1], reverse=True)]
            # print orderedItems
            # populate tree with ordered freq itemset
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable  # return tree and header table


def updateTree(items, inTree, headerTable, count):
    # check if orderedItems[0] in retTree.children
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)  # incease count
    else:  # add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:  # update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:  # update header table
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    # Do not use recursion to traverse a linked list!
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    '''Ascends from leaf node to root.'''
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)  # iteration


def findPrefixPath(basePat, treeNode):
    '''Use header table to generate conditional pattern base.
    TreeNode comes from header table.
    '''
    condPats = {}
    while treeNode is not None:
        prefixPath = []  # prefix path
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''Generate conditional FP-tree.'''
    # sort header table(for small to large)
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in bigL:  # start from the bottom of header table
        newFreqSet = preFix.copy()  # set
        newFreqSet.add(basePat)
        # print 'Final Frequent Item: ', newFreqSet  # append to set
        freqItemList.append(newFreqSet)  # list
        # generate conditional pattern base
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print 'Conditional Pattern Bases : ', basePat, condPattBases
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print 'Head from Conditional Tree: ', myHead
        if myHead is not None:  # 3. mine cond. FP-tree
            # print 'conditional tree for: ', newFreqSet
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
    # return freqItemList  # frequent item set


if __name__ == '__main__':
    import pprint
    parsedData = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedData)
    myTree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    mineTree(myTree, myHeaderTab, 100000, set([]), myFreqList)
    pprint.pprint(myFreqList)
