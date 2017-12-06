# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:19:01 2017

@author: libing

If you shut the door to all errors, truth will be shut out.

boosting: AdaBoost algorithm
"""
import numpy as np


def load_simp_data():
    '''A simple home_made data set.'''
    data_mat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    class_labels = [1., 1., -1.0, -1.0, 1.0]
    return data_mat, class_labels


def load_data_set(file_name):
    '''General function to parse tab-delimited floats.'''
    # get the number of fields
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    data_lab = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        data_lab.append(float(cur_line[-1]))
    return data_mat, data_lab


def stump_classify(data_mat, dimension, thresh_val, thresh_ineq):
    '''Classify the data by thresh_val and thresh_inequal.

    Datum at same side will be classified as -1, which decided on thresh_value
    and thresh_ineq, the other is +1. This classification can be done based
    on any feature(dimension).

    Args:
        data_mat: data set, a matrix
        dimension: feature index
        thresh_val: float, thresh value
        thresh_ineq: str, thresh inequal, less than or greater than
    Returns:
        an array, class labels
    '''
    ret_arr = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_arr[data_mat[:, dimension] <= thresh_val] = -1.0
    else:
        ret_arr[data_mat[:, dimension] > thresh_val] = -1.0
    return ret_arr


def build_stump(data_arr, class_labels, D):
    '''Find the best decision stump.
    Args:
        data_arr: data set
        class_labels: class labels
        D: weight vector
    Returns:
        the best decision stump, minimum error and the best class
    '''
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf  # init error sum, to +definity
    for i in range(n):  # loop over all dimensions
        range_min = np.min(data_mat[:, i])
        range_max = np.max(data_mat[:, i])
        step_size = (range_max - range_min) / num_steps
        # loop over all range in current dimension
        for j in range(-1, int(num_steps)+1):
            for inequal in ['lt', 'rt']:  # go over less than and greater than
                thresh_val = range_min + float(j)*step_size
                # call stump classify
                predict_val = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predict_val == label_mat] = 0
                # calculate total error multiplied by D
                weighted_error = float(D.T * err_arr)
                # print("Split: dim %d, thresh %.2f, thresh ineqal: %s, the \
                      # weighted error is %.3f" %
                      # (i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:  # update
                    min_error = weighted_error
                    best_class_est = predict_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['inequal'] = inequal
    return best_stump, min_error, best_class_est


def adaBoost(data_arr, class_labels, num_iter=40):
    '''adaBoost algorithm(Adaptive boosting).
    Args:
        data_arr: data set
        class_labels: data set labels
        num_iter: the number of iterations
    Returns:
        a list containing weak learners.
    '''
    weak_class_arr = []  # strong learner
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1))/m)  # init weight vector to all equal
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        # basic learner
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        # print("D: ", D.T)
        # calculate alpha, throw in max(error,eps) to account for error=0
        alpha = 0.5 * np.log((1.0-error)/max(error, 1e-16))
        best_stump['alpha'] = alpha
        # store stump parameters in array
        weak_class_arr.append(best_stump)
        # print('class_est: ', class_est)
        # update weight vector
        # exponent for D calculation, getting messy
        expon = np.multiply(-1*alpha*np.mat(class_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)
        # calculate training error of all classifiers, if this is 0 quit for
        # loop early(use break)
        agg_class_est += alpha*class_est
        # print('agg_class_est: ', agg_class_est)
        agg_errors = np.multiply(np.sign(agg_class_est) !=
                                 np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print('total error: ', error_rate, '\n')
        # exit if error_rate is equal to 0
        if error_rate == 0.0:
            break
    return weak_class_arr  # exit if maximum iterations


def ada_classify(data2class, classifier_arr):
    '''Use weak learners trained to do classification.
    Args:
        data2class: data set to classify
        classifier_arr: a list of weak learners
    Returns:
        a label matrix
    '''
    data_mat = np.mat(data2class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):  # loop over all weak learners
        # call stump classify
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['inequal'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)
