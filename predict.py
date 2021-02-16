#!/usr/bin/env python
# encoding: utf-8
# File Name: predict.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/17 21:57
import os
import pickle as pkl
import numpy as np
import scipy.io
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# logger = logging.getLogger(__name__)
def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
#         print(type(i), num_label.shape, num_label[i])
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred
def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
#         print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
#         logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
#     logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
#     logger.info("Average micro %.2f, Average macro %.2f",
#             np.mean(micro) * 100,
#             np.mean(macro) * 100)
    return np.mean(micro) * 100, np.mean(macro) * 100
