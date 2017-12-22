# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:47:28 2017

@author: libing

"""


import numpy as np


def load_dataset1():
    '''A simple dataset.'''
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def load_dataset2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def eculidean_sim(in_a, in_b):
    '''Calculate eculidean similarity of two column vectors: a and b.
    similarity = 1.0 / (1.0 + eculidean_distance(in_a, in_b))'''
    return 1.0 / (1.0 + np.linalg.norm(in_a-in_b))


def pearson_sim(in_a, in_b):
    '''Calculate pearson similarity of two column vectors: a and b.
    The return of np.corrcoef is in [-1, +1], so we use 0.5+0.5*np.corrcoef to
    normalize it to [0, +1].'''
    if len(in_a) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(in_a, in_b, rowvar=0)[0][1]


def cos_sim(in_a, in_b):
    '''Calculate cosine similarity of two column vectors: a and b.
    Calculate cosine of the angle between two vectors. If the angle is 90
    degree, the similarity is 0; if two vector have same directions, the
    similarity is 1.
    The return of cosine is in [-1, +1], so we use 0.5+0.5*cosine to
    normalize it to [0, +1].'''
    num = float(in_a.T*in_b)
    denom = np.linalg.norm(in_a) * np.linalg.norm(in_b)
    return 0.5 + 0.5*(num/denom)


def stand_est(data_mat, user, item, sim_meas):
    '''Score based on item similarity.
    Args:
        data_mat: np.matrix
            dataset
        item: int
            item to score
        sim_meas: func, cos_sim(default)/pearson_sim/eculidean_sim
            measure to calculate similarity
    Returns:
        float, the score of item
    '''
    n = data_mat.shape[1]  # the number of items
    sim_total = 0.0
    rat_sim_total = 0.0
    for i in range(n):
        user_rating = data_mat[user, i]
        if user_rating == 0:
            continue
        overlap = np.nonzero(np.logical_and(data_mat[:, item].A > 0,
                                            data_mat[:, i].A > 0))
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(data_mat[overlap, item],
                                  data_mat[overlap, i])
        print('the %d and %d similarity is %f' % (item, i, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total/sim_total


def svd_est(data_mat, user, item, sim_meas):
    '''Score based on SVD.
    Args:
        data_mat: np.matrix
            dataset
        item: int
            item to score
        sim_meas: func, cos_sim(default)/pearson_sim/eculidean_sim
            measure to calculate similarity
    Returns:
        float, the score of item
    '''
    n = data_mat.shape[1]
    a = 0
    sim_total = 0.0
    rat_sim_total = 0.0
    u, sigma, v = np.linalg.svd(data_mat)
    sigma2 = sigma ** 2
    sigma2_sum = sigma2.sum()
    for i in sigma2.cumsum():
        if i > sigma2_sum*0.9:  # retain 90% of the info
            a = sigma2.cumsum().tolist().index(i) + 1
            break
    sig_a = np.mat(np.eye(a) * sigma[:a])   # build diagonal matrix
    # create transformed items
    x_formed_items = data_mat.T * u[:, :a] * sig_a.I
    for j in range(n):
        user_rating = data_mat[user, j]
        if (user_rating == 0) or (j == item):
            continue
        similarity = sim_meas(x_formed_items[item, :].T,
                              x_formed_items[j, :].T)
        print('the %d and %d similarity is %f' % (item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total/sim_total


def recommend(data_mat, user, N=3, sim_meas=cos_sim, est_method=stand_est):
    '''Recommend system.
    Args:
        data_mat: np.matrix
            dataset
        N: int, default is 3
            top N to recommend
        sim_meas: func, cos_sim(default)/pearson_sim/eculidean_sim
            measure to calculate similarity
        est_method: func, default is stand_est
            estimate method
    Returns:
        a list containing recommends
        '''
    # look for unrated items
    unrated_items = np.nonzero(data_mat[user, :].A == 0)
    if len(unrated_items) == 0:
        print('you rated all items.')
    item_score = []
    for item in unrated_items:
        estimate_score = est_method(data_mat, user, item, sim_meas)  # rate
        item_score.append((item, estimate_score))
    # return top N unrated items
    return sorted(item_score, key=lambda jj: jj[1], reverse=True)[:N]


def print_mat(in_mat, thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(in_mat[i, j]) > thresh:
                print(1)
            else:
                print(0)
        print('')


def img_compress(num_sv=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        current_line = line.strip('\n')
        new_row = []
        for i in range(len(current_line)):
            new_row.append(int(current_line[i]))
        myl.append(new_row)
    mydata = np.mat(myl)
    print("========origin matrix========")
    print(mydata, thresh)
    u, sigma, v = np.linalg.svd(mydata)
    sig_recon = np.mat(np.zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k, k] = sigma[k]
    recon_mat = u[:, :num_sv] * sig_recon * v[:num_sv, :]
    print('========reconstructed matrix using %d singular values========' %
          num_sv)
    print(recon_mat, thresh)
