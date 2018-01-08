# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 10:15:53 2017

@author: libing
"""
import numpy as np


def load_dataset(filename):
    '''将文本文件导入到一个列表中'''
    data = []
    fr = open(filename)  # 路径
    for line in fr.readlines():
        cur_line = line.rstrip().split('\t')
        flt_line = list(map(float, cur_line))  # 函数式编程
        data.append(flt_line)
    return np.mat(data)


def dist_eclud(vec_A, vec_B):
    '''Euclidean distance'''
    return np.sqrt(np.sum(np.power((vec_A - vec_B), 2)))


def rand_cent(dataset, k):
    '''该函数为给定数据集构建一个包含k个随机质心的集合。
        随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小值
        和最大值来完成。然后生成0到0.1之间的随机数并通过取值范围和最小值，以便确保
        随机点数在数据的边界之内。
    '''
    n = np.shape(dataset)[1]  # 每个数据的维数
    centroids = np.mat(np.zeros((k, n)))  # k个质心，matrix形式
    for j in range(n):
        minJ = min(dataset[:, j])  # dataSet中第j列的最小值
        rangeJ = float(max(dataset[:, j]) - minJ)  # dataSet中第j列数据的差值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataset, k, dist_means=dist_eclud, create_cent=rand_cent):
    m = np.shape(dataset)[0]  # 数据集大小
    # 创建矩阵来存储每个点的簇分配结果，一列记录簇索引值，一列存储误差
    # 这里的误差是指当前点到簇质心的距离
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroids = create_cent(dataset, k)  # k个质心
    cluster_changed = True  # 迭代标识变量
    while cluster_changed:
        cluster_changed = False
        for i in range(m):  # 对数据集中的每一个数据
            min_dict = np.inf  # 无穷大
            min_index = -1
            for j in range(k):  # 数据与每个质心比较，找出距离最近的质心
                distJI = dist_means(centroids[j, :], dataset[i, :])  # 计算距离
                if distJI < min_dict:  # 寻找距离最近的质心
                    min_dict = distJI
                    min_index = j  # 该数据归为这个质心(簇)
            # 簇分配结果发生变化，更新迭代标识
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dict**2
        print(centroids)
        for cent in range(k):
            # 获得给定簇的所有点
            ptsInClust = dataset[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            # 计算该簇下所有点的平均值，更新质心
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, cluster_assment


def biKmeans(dataset, k, dist_meas=dist_eclud):
    '''为克服k-means算法收敛于最小值的算法。提出二分K-均值(bisecting K-means)算法。
        该算法首先将所有点作为一个簇，然后将簇一分为二。之后选择其中一个簇继续
        进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低误差平方和的值。
        基于误差平方和(SSE)的划分过程不断重复，直到得到用户指定的簇数目为止。
            将所有点看成一个簇
            当簇数目小于k时
                对于每一个簇
                    计算总误差
                    在给定簇的上k-means clustering(k=2)
                    计算将该簇一分为二之后的总误差
                选择使得总误差最小的那个簇进行划分操作
    '''
    m = np.shape(dataset)[0]  # 数据集中数据个数
    # 创建矩阵来存储每个点的簇分配结果，一列记录簇索引值，一列存储误差
    # 这里的误差是指当前点到簇质心的距离
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataset, axis=0).tolist()[0]  # 初始只有一个质心
    cent_list = [centroid0]  # 保留所有的质心
    # 遍历数据集中的所有点来计算每个点到质心的误差值
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(np.mat(centroid0), dataset[j, :]) ** 2
    while (len(cent_list) < k):
        lowest_SSE = np.inf
        for i in range(len(cent_list)):  # 尝试划分每一簇
            # 每一簇里含有的所有数据
            ptsInCurrCluster = \
                dataset[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            # 对该簇进行k-means聚类(k=2)
            centroidMat, splitClustAss = \
                kMeans(ptsInCurrCluster, 2, dist_meas)
            sseSplit = np.sum(splitClustAss[:, 1])  # 该簇划分后sse
            # 剩余数据sse
            sseNotSplit = \
                np.sum(cluster_assment[np.nonzero(
                                        cluster_assment[:, 0].A != i)[0], 1])
            print('sseSplit, and sseNotSplit: ', sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowest_SSE:  # 若划分后sse减小
                bestCentToSplit = i                   # 最佳的划分簇
                bestNewCents = centroidMat            # 划分后质心
                bestClustAss = splitClustAss.copy()
                lowest_SSE = sseSplit + sseNotSplit    # 划分后总sse
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = \
            len(cent_list)  # 进行划分操作，簇划分后类别为1的更新为新加簇编号
        bestClustAss[np.nonzero(bestClustAss[:, 1].A == 0)[0], 0] = \
            bestCentToSplit  # 簇划分后类别为0的更新为原划分簇的编号
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新新的簇分配结果
        cent_list[bestCentToSplit] = bestNewCents[0, :]
        cent_list.append(bestNewCents[1, :])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A ==
                                   bestCentToSplit)[0], :] = bestClustAss
    return np.mat(cent_list), cluster_assment
