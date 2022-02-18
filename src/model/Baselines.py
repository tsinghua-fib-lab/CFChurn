import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn.inits import glorot, zeros

import sys
sys.path.append('./')
from .layers import *


EPS = 1e-15


class FIN(nn.Module):
    """docstring for FIN"""
    def __init__(self, args):
        super(FIN, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_embedding = 10
        self.field_size = 8
        self.n_channels_c = args.n_channels_c
        self.n_users = args.num_targets

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.channel = self.nh0 - self.field_size + 1
        self.act = activation_layer(args.activation)

        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c, use_bn=False)

        self.v = nn.Parameter(torch.Tensor(1, self.channel, self.field_size, self.n_embedding))
        self.fm = FactorizationMachine(reduce_sum=True)
        self.lin = nn.Linear(self.field_size, 1)

        self.prediction = PredictionLayer(self.nh0 + self.channel, [int((self.nh0 + self.channel)/2)], 'leaky_relu')

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.v)

    def forward(self, data):
        discrete_x, continous_x = data.discrete_x, data.continous_x

        discrete_x = discrete_x[:self.n_users, :]
        continous_x = continous_x[:self.n_users, :]

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x = torch.cat([x_d, x_c], dim=-1)  # batch * nh0

        x_fm = x.reshape(x.size(0), 1, 1, -1)
        x_fm = F.unfold(x_fm, (1, self.field_size), 1, 0, 1)  # batch * field_size (5) * channel (即滑窗多少次，最后应该输出batch * channel)
        x_fm = x_fm.permute(0, 2, 1).unsqueeze(-1)  # batch * channel * field_size * 1
        x_fm_v = x_fm * self.v
        x_fm_v = self.fm(x_fm_v)  # batch * channel * 1
        x_fm = self.lin(x_fm.squeeze(-1)) + x_fm_v
        x_fm = x_fm.squeeze(-1)

        res = torch.cat([x, x_fm], dim=-1)
        pred_y = self.prediction(res)
        pred_y = torch.sigmoid(pred_y)

        return pred_y


class FIN_b(nn.Module):
    """docstring for FIN_b"""
    def __init__(self, args):
        super(FIN_b, self).__init__()
        self.args = args
        self.ndf = args.n_discrete_features
        self.ncf = args.n_continous_features
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.field_size = 8
        self.ncc = args.n_channels_c
        self.n_users = args.num_targets

        self.nh0 = self.ndf + 2 * self.ncc
        self.channel = self.nh0 - self.field_size + 1
        self.activation = args.activation
        self.act = activation_layer(args.activation)

        self.embedding_d = nn.Linear(self.ndf, self.ndf)
        self.embedding_c = CEmbedding(self.ncf, self.ncc, use_bn=False)

        self.v = nn.Parameter(torch.Tensor(1, self.channel, self.field_size, self.n_embedding))
        self.fm = FactorizationMachine(reduce_sum=True)
        self.lin = nn.Linear(self.field_size, 1)

        self.prediction = PredictionLayer(self.nh0 + self.channel, [int((self.nh0 + self.channel)/2)], 'leaky_relu')

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.v)

    def forward(self, data):
        discrete_x, continous_x = data.discrete_x, data.continous_x

        discrete_x = discrete_x[:self.n_users, :]
        continous_x = continous_x[:self.n_users, :]

        x_d = self.act(self.embedding_d(discrete_x))
        x_c = self.act(self.embedding_c(continous_x))
        x = torch.cat([x_d, x_c], dim=-1)  # batch * nh0

        x_fm = x.reshape(x.size(0), 1, 1, -1)
        x_fm = F.unfold(x_fm, (1, self.field_size), 1, 0, 1)  # batch * field_size (5) * channel (即滑窗多少次，最后应该输出batch * channel)
        x_fm = x_fm.permute(0, 2, 1).unsqueeze(-1)  # batch * channel * field_size * 1
        x_fm_v = x_fm * self.v
        x_fm_v = self.fm(x_fm_v)  # batch * channel * 1
        x_fm = self.lin(x_fm.squeeze(-1)) + x_fm_v
        x_fm = x_fm.squeeze(-1)

        res = torch.cat([x, x_fm], dim=-1)
        pred_y = self.prediction(res)
        pred_y = torch.sigmoid(pred_y)

        return pred_y


class BaseGAT(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(BaseGAT, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = args.n_channels_c
        self.heads = args.heads
        self.n_channels_d = 6

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden
        self.activation = args.activation
        self.act = activation_layer(args.activation)

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GATConv(self.n_hidden, self.n_hidden, heads=1, concat=False)
        self.embeding_g2 = GATConv(self.n_hidden, self.n_hidden, heads=1, concat=False)

        # prediction
        self.y_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x_g = F.dropout(x_g, p=self.dropout, training=self.training)

        pred_y = self.y_predicter(x_g)

        return pred_y


class BaseGCN(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(BaseGCN, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = args.n_channels_c
        self.heads = args.heads
        self.n_channels_d = 6

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden
        self.activation = args.activation
        self.act = activation_layer(args.activation)

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        # prediction
        self.y_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x_g = F.dropout(x_g, p=self.dropout, training=self.training)

        pred_y = self.y_predicter(x_g)

        return pred_y


class BaseGCN_b(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(BaseGCN_b, self).__init__()
        self.args = args
        self.ndf = args.n_discrete_features
        self.ncf = args.n_continous_features
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.ncc = args.n_channels_c

        self.nh0 = self.ndf + 2 * self.ncc
        self.nh1 = self.nh0 + self.n_hidden
        self.activation = args.activation
        self.act = activation_layer(args.activation)

        # embeddng
        self.embedding_d = nn.Linear(self.ndf, self.ndf)
        self.embedding_c = CEmbedding(self.ncf, self.ncc)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        # prediction
        self.y_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index = \
            data.discrete_x, data.continous_x, data.edge_index

        x_d = self.act(self.embedding_d(discrete_x))
        x_c = self.act(self.embedding_c(continous_x))
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x_g = F.dropout(x_g, p=self.dropout, training=self.training)

        pred_y = self.y_predicter(x_g)

        return pred_y

