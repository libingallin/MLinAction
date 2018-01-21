# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:44:25 2018

@author: libing

The usual covariance maximum likelihood estimate can be regularized using
shrinkage. Ledoit and Wolf proposed a close formula to compute the
asymptotically optimal shrinkage parameter(minimizing a MSE criterion),
yielding the Ledoit-Wolf covariance estimate.

Chen et al. proposed an improvement of the Ledoit-Wolf shrinkage parameter, the
OAS coefficient, whose convergence is significantly better under the assumption
that the data are Gaussian.

This example shows a comparsion of the estimated MSE of the LW and OAS methods,
using Gaussion distributed data.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky

from sklearn.covariance import LedoitWolf, OAS


n_features = 100
# simulation covariacne matrix (AR(1) process)
r = 0.1
real_cov = toeplitz(r ** np.arange(n_features))
coloring_matrix = cholesky(real_cov)

n_samples_range = np.arange(6, 31, 1)
repeat = 100
lw_mse = np.zeros((n_samples_range.size, repeat))
oas_mse = np.zeros((n_samples_range.size, repeat))
lw_shrinkage = np.zeros((n_samples_range.size, repeat))
oas_shrinkage = np.zeros((n_samples_range.size, repeat))
for i, n_samples in enumerate(n_samples_range):
    for j in range(repeat):
        X = np.dot(np.random.normal(size=(n_samples, n_features)),
                   coloring_matrix)
        lw = LedoitWolf(store_precision=False, assume_centered=True)
        lw.fit(X)
        lw_mse[i, j] = lw.error_norm(real_cov, scaling=False)
        lw_shrinkage[i, j] = lw.shrinkage_

        oas = OAS(store_precision=False, assume_centered=True)
        oas.fit(X)
        oas_mse[i, j] = oas.error_norm(real_cov, scaling=False)
        oas_shrinkage[i, j] = oas.shrinkage_

# plot MSE
plt.subplot(211)
plt.errorbar(n_samples_range, lw_mse.mean(1), yerr=lw_mse.std(1),
             label='Ledoit-Wolf', color='navy', lw=2)
plt.errorbar(n_samples_range, oas_mse.mean(1), yerr=oas_mse.std(1),
             label='OAS', color='darkorange', lw=2)
plt.ylabel('Squared error')
plt.legend()
plt.title('Comparsion of covariance estimators')
plt.xlim(5, 31)

# plot shrinkage coefficient
plt.subplot(212)
plt.errorbar(n_samples_range, lw_shrinkage.mean(1), yerr=lw_shrinkage.std(1),
             label='Ledoit-Wolf', color='navy', lw=2)
plt.errorbar(n_samples_range, oas_shrinkage.mean(1), yerr=oas_shrinkage.std(1),
             label='OAS', color='darkorange', lw=2)
plt.legend()
plt.xlabel('n_samples')
plt.ylabel('Shrinkage')
plt.ylim(plt.ylim()[0], 1.0+(plt.ylim()[1]-plt.ylim()[0])/10.0)
plt.xlim(5, 31)

plt.show()
