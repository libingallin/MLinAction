# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:43:26 2018

@author: libing

Pipelining: chaining a PCA(unsupervised dimensionality reduction) and a
logistic regreession(does the prection).

Use a GridSearchCV to set the dimensionality of the PCA.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize=(4, 3))
plt.axes([0.2, 0.2, 0.7, 0.7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance')

# prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# parameters of pipelines can be set using '__' separated parameter names:
estimator = GridSearchCV(
        pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
