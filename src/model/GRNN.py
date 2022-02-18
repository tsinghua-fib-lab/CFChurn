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


class GRNN(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(GRNN, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.dropout = args.dropout
        self.num_nodes = args.num_nodes
        self.n_channels_c = args.n_channels_c
        self.n_layers = 1

        self.nh0 = self.n_discrete_features - 6 + self.n_channels_c
        self.activation = args.activation
        self.act = activation_layer(args.activation)
        self.h0 = torch.zeros(self.n_layers, self.num_nodes, self.n_channels_c)
        self.h0 = self.h0.to(args.device)

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = RNNEmbeddingLayer(
            self.n_continous_features, self.n_channels_c, self.n_layers)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        # prediction
        self.y_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index = \
            data.discrete_x, data.continous_x, data.edge_index

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x, self.h0)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = x_g + x_g0 + x_g1
        x = F.dropout(x, p=self.dropout, training=self.training)

        pred_y = self.y_predicter(x)

        return pred_y


class GRNN2(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(GRNN2, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.dropout = args.dropout
        self.num_nodes = args.num_nodes
        self.n_channels_c = args.n_channels_c
        self.n_layers = 2

        self.nh0 = self.n_discrete_features - 6 + self.n_channels_c
        self.activation = args.activation
        self.act = activation_layer(args.activation)
        self.h0 = torch.zeros(self.n_layers, self.num_nodes, self.n_channels_c)
        self.h0 = self.h0.to(args.device)

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = RNNEmbeddingLayer(
            self.n_continous_features, self.n_channels_c, self.n_layers)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        # prediction
        self.y_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index = \
            data.discrete_x, data.continous_x, data.edge_index

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x, self.h0)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x_g = F.dropout(x_g, p=self.dropout, training=self.training)

        pred_y = self.y_predicter(x_g)

        return pred_y
