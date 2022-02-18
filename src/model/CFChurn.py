import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sys
sys.path.append('./')
from .layers import *


class CFChurn_DNN_SGAT(nn.Module):
    '''
    在CFChurn的基础上，去掉DCN，只用feature + DNN，看效果怎么样
    '''
    def __init__(self, args):
        super(CFChurn_DNN_SGAT, self).__init__()
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
        self.embedding_e = EdgeLearning(self.n_hidden, self.n_edge_features,
                                        self.n_hidden, self.activation)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden])

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = EGATConv(self.n_hidden, self.n_hidden, self.n_hidden, heads=self.heads)
        self.social_inf_g2 = EGATConv(self.n_hidden, self.n_hidden, self.n_hidden, heads=self.heads)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t, edge_index_alive, \
            edge_attr_alive = data.discrete_x, data.continous_x, \
            data.edge_index, data.edge_attr, data.t, data.edge_index_alive, \
            data.edge_attr_alive

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        h_ci = F.dropout(x_deep, p=self.dropout, training=self.training)
        h_ci = self.fusion_ci(h_ci)

        # social influence
        x_si = self.act(self.social_inf_g0(x))
        edge_attr = self.embedding_e(x_si, edge_index, edge_attr)
        x_si0 = self.act(self.social_inf_g1(x_si, edge_index, edge_attr))
        x_si1 = self.act(self.social_inf_g2(x_si0, edge_index, edge_attr))
        h_si = x_si0 + x_si1
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)

        pred_T = self.T_predicter(h_si)

        h = torch.cat([h_ci, h_si], dim=-1)
        alpha_y0 = self.attn_y0(h)
        alpha_y0 = torch.softmax(alpha_y0, dim=-1)
        pred_y0 = alpha_y0[:, :self.n_hidden] * h_ci + alpha_y0[:, self.n_hidden:] * h_si

        alpha_y1 = self.attn_y1(h)
        alpha_y1 = torch.softmax(alpha_y1, dim=-1)
        pred_y1 = alpha_y1[:, :self.n_hidden] * h_ci + alpha_y1[:, self.n_hidden:] * h_si

        pred_y0 = self.y0_predicter(pred_y0)
        pred_y1 = self.y1_predicter(pred_y1)

        pred_T = pred_T[:len(t), :]
        pred_y0 = pred_y0[:len(t), :]
        pred_y1 = pred_y1[:len(t), :]

        pred_y = (1 - t) * pred_y0 + t * pred_y1
        pred_y_cf = t * pred_y0 + (1 - t) * pred_y1

        return (pred_y, pred_y_cf, pred_y0, pred_y1, pred_T, h_ci, h_si)