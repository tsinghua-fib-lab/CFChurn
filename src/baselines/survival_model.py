# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:36:53 2018

@author: zgz
"""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test

import numpy as np
import pandas as pd
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score


n_folds = 5
fold = 0
seed = 666


def split_train_val_test(n, nfolds, seed):
    '''
    n-fold cross validation
    '''
    train_idx, valid_idx, test_idx = {}, {}, {}
    rnd_state = np.random.RandomState(seed)
    idx_all = rnd_state.permutation(n)
    idx_all = idx_all.tolist()
    stride = int(n / (nfolds + 1))
    # 先把idx分成n + 1份;最后的1份是test_set
    idx = [idx_all[(i + 1) * stride:(i + 2) * stride] for i in range(nfolds)]
    for fold in range(nfolds):
        valid_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i != fold:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
        test_idx[fold] = np.array(copy.deepcopy(idx_all[:stride]))
    return train_idx, valid_idx, test_idx


def evaluation(pred_y, y):
    TP = ((pred_y == 1) * (y == 1)).sum()
    TN = ((pred_y == 0) * (y == 0)).sum()
    FN = ((pred_y == 0) * (y == 1)).sum()
    FP = ((pred_y == 1) * (y == 0)).sum()

    precision = 0
    recall = 0
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    auc = roc_auc_score(y, pred_y)
    return precision, recall, accuracy, auc


dataset = np.load('sample12_dataset_norm.npy', allow_pickle=True)

discrete_x = np.array(dataset[0])
continous_x = np.array(dataset[1])
labels = np.array([list(u) for u in dataset[4]])
y = (labels[:, 1] > 0).astype(np.int32)
t = (labels[:, 0] > 0).astype(np.int32)
num_targets = int(y.shape[0])
discrete_x = discrete_x[:num_targets, [0, 1, 4, 12, 13, 14, 16, 17, 18, 19]]
continous_x = continous_x[:num_targets, :]


# 利用continuousx还原出一个人到底是第几周流失的：如果y是0，那么就不用管
duration = []
for i in range(y.shape[0]):
    if y[i] == 0:
        duration.append(13)
    else:
        d = continous_x[1, :13] + continous_x[1, 13:26] + continous_x[1, 26:39]
        for j in range(13):
            if d[12 - j] > 0:
                break
        count = 13 - j
        duration.append(count)
duration = np.array(duration).reshape(-1, 1)

duration_pos = discrete_x.shape[1] + 2
event_pos = discrete_x.shape[1] + 1
survival_data = np.concatenate((discrete_x, t.reshape(-1, 1), y.reshape(-1, 1),
                                duration), axis=1)

train_idx, val_idx, test_idx = split_train_val_test(num_targets, n_folds, seed)
train_idx, val_idx, test_idx = train_idx[fold], val_idx[fold], test_idx[fold]
train_idx = np.concatenate((train_idx, val_idx), axis=0)
train_data = survival_data[train_idx]
test_data = survival_data[test_idx]

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

cph = CoxPHFitter()
cph.fit(train_data, duration_col=duration_pos, event_col=event_pos)
cph.print_summary()

test_data = test_data[:][[i for i in range(discrete_x.shape[1] + 1)]]

best_threshold = 0
best_auc = 0
best_acc = 0
best_p = 0
best_r = 0
for i in range(58, 73):
    threshold = i/100
    predict = cph.predict_percentile(test_data, threshold)
    pred_y = predict.to_numpy() < 13
    pred_y = pred_y.astype(int)

    precision, recall, accuracy, auc = evaluation(pred_y, y[test_idx])
    if auc > best_auc:
        best_threshold = threshold
        best_auc = auc
        best_acc = accuracy
        best_p = precision
        best_r = recall

print('best threshold ', best_threshold)
print('precision: ', best_p)
print('recall: ', best_r)
print('accuracy: ', best_acc)
print('auc: ', best_auc)
