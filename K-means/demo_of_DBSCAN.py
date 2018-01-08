# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:01:38 2018

@author: libing

Finds core samples of high density and expands clusters from them.

From sklearn.cluster.DBSCAN
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
# build artificial datasets of controlled size and complexity
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)

# compute DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

# number of clusters in labels, ignoring noise if present
n_cluster_ = len(set(labels)) - (1 if -1 in labels else 0)

# estimate clustering performance
print('Estimated number of clusters: %d' % n_cluster_)
print('Homogeneity: %0.3f\nConpleteness: %0.3f\nV_measure: %0.3f'
      % metrics.homogeneity_completeness_v_measure(labels_true, labels))
print('Adjusted Rand Index: %0.3f'
      % metrics.adjusted_rand_score(labels_true, labels))
print('Adjusted Mutual Information: %0.3f'
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, labels))

# plot result
# black removed and is used for noise instead
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # black used for noise
    class_member_mask = (labels == k)
    # plot core samples
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    # plot non_core samples
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_cluster_)
plt.show()
