# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import torch
import torch.utils
import torch.utils.data
import numpy as np
import copy

from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)


def create_dataset(path):
    # [discrete_x_matrix, continous_x_matrix, edges, edge_attr_matrix, churn_x
    #            label_matrix]

    data_samples = np.load(path, allow_pickle=True)

    discrete_x = torch.tensor(data_samples[0], dtype=torch.float)
    discrete_x = discrete_x[:, :16]
    continous_x = torch.tensor(data_samples[1], dtype=torch.float)
    edge_index = torch.tensor(data_samples[2], dtype=torch.long)

    edge_attr = torch.tensor(data_samples[3], dtype=torch.float)
    print('edge_attr shape: ', edge_attr.size())

    edge_attr_sum = torch.sum(edge_attr, dim=1)
    edge_attr_alive = edge_attr[edge_attr_sum != 0]
    edge_index_alive = edge_index[edge_attr_sum != 0]

    edge_index = edge_index.t().contiguous()
    print('edge_index shape: ', edge_index.size())
    edge_index_alive = edge_index_alive.t().contiguous()
    print('edge_index_alive shape: ', edge_index_alive.size())
    print('edge_attr_alive shape: ', edge_attr_alive.size())

    labels = torch.tensor([list(u) for u in data_samples[4]], dtype=torch.float)
    y = (labels[:, 1] > 0).float().view(-1, 1)
    t = (labels[:, 0] > 0).float().view(-1, 1)
    print('y shape: ', y.size())
    # treatment = torch.tensor(data_samples[5], dtype=torch.float)
    # churn_date = torch.tensor(data_samples[6], dtype=torch.float)
    tem = torch.tensor(data_samples[5], dtype=torch.float)
    treatment = tem[:, :1]
    churn_date = tem[:, 1:]

    churn_date = churn_date - 364
    churn_date[churn_date < 0] = 183
    churn_date[churn_date > 183] = 183
    churn_date = 1 - churn_date / 183

    y_cf = ((y == 1) * (t == 0)).float().view(-1, 1)
    id_cf = ((y == 1) * (t == 0) + (y == 0) * (t == 1)).float().view(-1, 1)

    # 给t = 1加了一堆会流失的人；给t = 0加了一堆不会流失的人；但问题是，其实t = 1才更不容易流失= =
    print('#cf_t1 & t0_y1: ', torch.sum((y == 1) * (t == 0)))
    print('#cf_t0 & t1_y0: ', torch.sum((y == 0) * (t == 1)))
    print('#t0: ', torch.sum(t == 0))
    print('#t1: ', torch.sum(t == 1))

    pos_edge_index, _ = remove_self_loops(edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    neg_edge_index = negative_sampling(pos_edge_index, discrete_x.size(0))

    dataset = Data(discrete_x=discrete_x, continous_x=continous_x,
                   edge_index=edge_index, edge_attr=edge_attr,
                   treatment=treatment, y=y, t=t, y_cf=y_cf, id_cf=id_cf,
                   neg_edge_index=neg_edge_index, churn_date=churn_date,
                   edge_attr_alive=edge_attr_alive,
                   edge_index_alive=edge_index_alive)
    return dataset


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


def make_batch(train_ids, batch_size, seed):
    """
    return a list of batch ids for mask-based batch.
    Args:
        train_ids: list of train ids
        batch_size: ~
    Output:
        batch ids, e.g., [[1,2,3], [4,5,6], ...]
    """

    num_nodes = len(train_ids)
    rnd_state = np.random.RandomState(seed)
    permuted_idx = rnd_state.permutation(num_nodes)
    permuted_train_ids = train_ids[permuted_idx]
    batches = [permuted_train_ids[i * batch_size:(i + 1) * batch_size] for
               i in range(int(num_nodes / batch_size))]
    if num_nodes % batch_size > 0:
        if (num_nodes % batch_size) > 0.5 * batch_size:
            batches.append(
                permuted_train_ids[(num_nodes - num_nodes % batch_size):])
        else:
            batches[-1] = np.concatenate((batches[-1], permuted_train_ids[(num_nodes - num_nodes % batch_size):]))

    return batches, num_nodes
