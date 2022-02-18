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


class CFChurn_DNN(nn.Module):
    '''
    在CFChurn的基础上，去掉DCN，只用feature + DNN，看效果怎么样
    '''
    def __init__(self, args):
        super(CFChurn_DNN, self).__init__()
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
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

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


class CFChurn_DNN_SGCN(nn.Module):
    '''
    在CFChurn的基础上，去掉DCN，只用feature + DNN，看效果怎么样
    '''
    def __init__(self, args):
        super(CFChurn_DNN_SGCN, self).__init__()
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
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

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
        edge_attr = self.embedding_e(x_si, edge_index_alive, edge_attr_alive)
        x_si0 = self.act(self.social_inf_g1(x_si, edge_index_alive, edge_attr))
        x_si1 = self.act(self.social_inf_g2(x_si0, edge_index_alive, edge_attr))
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


class CFChurn_DNN2(nn.Module):
    '''
    在CFChurn_DNN的基础上，只保留最后几位continuous features，看效果怎么样
    - 效果不佳
    '''
    def __init__(self, args):
        super(CFChurn_DNN2, self).__init__()
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
        self.embedding_c = ContinousFeatureEmbeddingLayer2(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)
        self.embedding_e = EdgeLearning(self.n_hidden, self.n_edge_features,
                                        self.n_hidden, self.activation)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=True,
                             dropout_rate=self.dropout)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

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



class CFChurn10(nn.Module):
    """
    在cfchurn7的基础上：
    1. gnn embedding的readout也改加法
    2. 用了ELConv
    """
    def __init__(self, args):
        super(CFChurn10, self).__init__()
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
                             self.activation, use_bn=True,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
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



class SimpleDNN(nn.Module):
    '''
    在CFChurn的基础上，去掉DCN，只用feature + DNN，看效果怎么样
    '''
    def __init__(self, args):
        super(SimpleDNN, self).__init__()
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

        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)

        self.dnn = DNN(self.nh0,
                       [self.n_hidden] * self.l_hidden, self.activation,
                       use_bn=self.bn, dropout_rate=self.dropout)
        self.prediction = PredictionLayer(self.n_hidden, [int(self.n_hidden/2)] ,self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x = torch.cat([x_d, x_c], dim=-1)
        x = self.dnn(x)
        pred_y = self.prediction(x).squeeze()

        return pred_y


class CFChurn_b(nn.Module):
    """
    在cfchurn7的基础上：
    1. gnn embedding的readout也改加法
    2. 用了ELConv
    """
    def __init__(self, args):
        super(CFChurn_b, self).__init__()
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
        self.embedding_e = EdgeLearning(self.n_hidden, self.n_edge_features,
                                        self.n_hidden, self.activation)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=True,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

        x_d = self.act(self.embedding_d(discrete_x))
        x_c = self.act(self.embedding_c(continous_x))
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
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

# --------------------------------------------------------


class CFChurn(nn.Module):
    """docstring for CounterfactualChurn"""
    def __init__(self, args):
        super(CFChurn, self).__init__()
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
        self.act = activation_layer(args.activation)

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)
        self.embedding_e = DNN(self.n_edge_features, [self.n_edge_features]*2, use_bn=True)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             'leaky_relu', use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = nn.Linear(2 * self.nh1, self.n_hidden)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)
        self.social_inf_g2 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)
        self.lin1_si = nn.Linear(self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden] * 2, 'leaky_relu')
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden] * 2, 'leaky_relu')
        self.T_predicter = PredictionLayer(self.n_hidden, [], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t= \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        edge_attr = self.embedding_e(edge_attr)

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = torch.cat([x_deep, x_cross], dim=-1)
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = self.act(self.fusion_ci(h_ci))

        # social influence
        x_si = self.act(self.social_inf_g0(x))
        x_si = self.act(self.social_inf_g1(x_si, edge_index, edge_attr))
        x_si = self.act(self.social_inf_g2(x_si, edge_index, edge_attr))
        x_si = F.dropout(x_si, p=self.dropout, training=self.training)
        h_si = self.act(self.lin1_si(x_si))

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


class CFChurn7(nn.Module):
    """
    在cfchurn6的基础上：
    1. 尽量去除不需要的线性层，缩短了prediction路径
    2. readout改加法了
    """
    def __init__(self, args):
        super(CFChurn7, self).__init__()
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
        self.embedding_e = DNN(self.n_edge_features, [self.n_edge_features]*2, use_bn=True)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)
        self.social_inf_g2 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t= \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        edge_attr = self.embedding_e(edge_attr)

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g = self.act(self.embeding_g1(x_g, edge_index))
        x_g = self.act(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = self.fusion_ci(h_ci)

        # social influence
        x_si = self.act(self.social_inf_g0(x))
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


class CFChurn11(nn.Module):
    """
    在cfchurn10的基础上：
    把用户流失时间作为一个维度加入，再加个头，看社交影响的的情况
    """
    def __init__(self, args):
        super(CFChurn11, self).__init__()
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
                             self.activation, use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # social influence new
        self.social_inf_c0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_c1 = GateGCN(self.n_hidden, self.n_hidden)
        self.social_inf_c2 = GateGCN(self.n_hidden, self.n_hidden)

        # tri attn
        self.attn_y0 = nn.Linear(3*self.n_hidden, 3*self.n_hidden)
        self.attn_y1 = nn.Linear(3*self.n_hidden, 3*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = self.fusion_ci(h_ci)

        # social influence
        x_si = self.act(self.social_inf_g0(x))
        edge_attr = self.embedding_e(x_si, edge_index, edge_attr)
        x_si0 = self.act(self.social_inf_g1(x_si, edge_index, edge_attr))
        x_si1 = self.act(self.social_inf_g2(x_si0, edge_index, edge_attr))
        h_si = x_si0 + x_si1
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)

        # social influence new
        x_ns = self.act(self.social_inf_c0(x))
        x_ns0 = self.act(self.social_inf_c1(x_ns, edge_index, churn_date))
        x_ns1 = self.act(self.social_inf_c1(x_ns0, edge_index, churn_date))
        h_ns = x_ns0 + x_ns1
        h_ns = F.dropout(h_ns, p=self.dropout, training=self.training)

        pred_T = self.T_predicter(h_si)

        h = torch.cat([h_ci, h_si, h_ns], dim=-1)
        alpha_y0 = self.attn_y0(h)
        alpha_y0 = torch.softmax(alpha_y0, dim=-1)
        pred_y0 = alpha_y0[:, :self.n_hidden] * h_ci + alpha_y0[:, self.n_hidden:2*self.n_hidden] * h_si + alpha_y0[:, 2*self.n_hidden:] * h_ns

        alpha_y1 = self.attn_y1(h)
        alpha_y1 = torch.softmax(alpha_y1, dim=-1)
        pred_y1 = alpha_y1[:, :self.n_hidden] * h_ci + alpha_y1[:, self.n_hidden:2*self.n_hidden] * h_si + alpha_y1[:, 2*self.n_hidden:] * h_ns

        pred_y0 = self.y0_predicter(pred_y0)
        pred_y1 = self.y1_predicter(pred_y1)

        pred_T = pred_T[:len(t), :]
        pred_y0 = pred_y0[:len(t), :]
        pred_y1 = pred_y1[:len(t), :]

        pred_y = (1 - t) * pred_y0 + t * pred_y1
        pred_y_cf = t * pred_y0 + (1 - t) * pred_y1

        return (pred_y, pred_y_cf, pred_y0, pred_y1, pred_T, h_ci, h_si)


class CFChurn12(nn.Module):
    """
    在cfchurn7的基础上：
    1. gnn embedding的readout也改加法
    2. 用了 EGAT4
    """
    def __init__(self, args):
        super(CFChurn12, self).__init__()
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
        self.embedding_e = DNN(self.n_edge_features, [self.n_edge_features]*2, use_bn=True)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)
        self.social_inf_g2 = EGATConv(self.n_hidden, self.n_hidden, self.n_edge_features, heads=self.heads)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t= \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        edge_attr = self.embedding_e(edge_attr)

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = self.fusion_ci(h_ci)

        # social influence
        x_si = self.act(self.social_inf_g0(x))
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


class CFChurn13(nn.Module):
    """
    在cfchurn10的基础上：
    1. 去掉了所有bn
    """
    def __init__(self, args):
        super(CFChurn13, self).__init__()
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
            self.n_continous_features, self.n_channels_c, self.bn)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)
        self.embedding_e = EdgeLearning(self.n_hidden, self.n_edge_features,
                                        self.n_hidden, self.activation)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
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


class CFChurn15(nn.Module):
    """
    在cfchurn10的基础上：
    1. 用了rnn做embedding
    """
    def __init__(self, args):
        super(CFChurn15, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = args.n_channels_c
        self.n_channels_d = 6
        self.num_nodes = args.num_nodes
        self.n_layers = 1

        self.nh0 = self.n_discrete_features - 6 + self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden
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
        self.embedding_e = EdgeLearning(self.n_hidden, self.n_edge_features,
                                        self.n_hidden, self.activation)

        # churn intention module
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             self.activation, use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = DNN(self.nh1, [self.n_hidden], use_bn=False)

        # social influence module
        self.social_inf_g0 = nn.Linear(self.nh1, self.n_hidden)
        self.social_inf_g1 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)
        self.social_inf_g2 = ELConv(self.n_hidden, self.n_hidden, self.n_hidden)

        # dual attn
        self.attn_y0 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)
        self.attn_y1 = nn.Linear(2*self.n_hidden, 2*self.n_hidden)

        self.y0_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.y1_predicter = PredictionLayer(self.n_hidden, [self.n_hidden], self.activation)
        self.T_predicter = PredictionLayer(self.n_hidden, [], self.activation)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, t = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.t

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x, self.h0)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = self.act(self.embeding_g0(x_g))
        x_g0 = self.act(self.embeding_g1(x_g, edge_index))
        x_g1 = self.act(self.embeding_g2(x_g0, edge_index))
        x = torch.cat([x_d, x_c, x_g0 + x_g1], dim=-1)

        # churn intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = x_deep + x_cross
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
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


class DCN(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_channels_c = 4

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden

        # self intention
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        self.dnn = DNN(self.nh1,
                       [self.nh1] * self.l_hidden, 'leaky_relu',
                       use_bn=self.bn, dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.prediction = PredictionLayer(2*self.nh1, [self.nh1], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = F.leaky_relu(self.embeding_g0(x_g))
        x_g = F.leaky_relu(self.embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        x_deep = self.dnn(x)
        x_cross = self.cross(x)
        x = torch.cat([x_deep, x_cross], dim=-1)

        pred_y = self.prediction(x)

        return pred_y


class ResDCN(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(ResDCN, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden

        self.nh = self.n_discrete_features - 6

        self.feature_embedding_d = DiscreteFeatureEmbeddingLayer()
        self.feature_embeding_g0 = nn.Linear(self.nh, self.n_embedding)
        self.feature_embeding_g1 = GCNConv(self.n_embedding, self.n_embedding)
        self.feature_embeding_g2 = GCNConv(self.n_embedding, self.n_embedding)

        self.resdnn = ResDNN(self.nh + self.n_embedding, self.l_hidden, 'leaky_relu',
                             use_bn=self.bn, dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh + self.n_embedding, 2)
        self.prediction = PredictionLayer(2 * (self.nh + self.n_embedding), [int((2*self.nh + self.n_hidden)/2)], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = self.feature_embedding_d(discrete_x)
        x_g = F.leaky_relu(self.feature_embeding_g0(x_d))
        x_g = F.leaky_relu(self.feature_embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.feature_embeding_g2(x_g, edge_index))

        x = torch.cat([x_d, x_g], dim=-1)
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        x = torch.cat([x_deep, x_cross], dim=-1)

        y_hat = self.prediction(x).squeeze()

        return y_hat


class Base_single_embedding(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(Base_single_embedding, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.dropout = args.dropout
        self.device = args.device

        self.feature_embeding_d = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        self.feature_embeding_c1 = nn.Linear(
            self.n_continous_features, self.n_embedding)
        self.feature_embeding_c2 = nn.Linear(
            self.n_continous_features, self.n_embedding)
        self.feature_embeding_g0 = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        self.feature_embeding_g1 = GCNConv(self.n_embedding, self.n_embedding)
        self.feature_embeding_g2 = GCNConv(self.n_embedding, self.n_embedding)
        self.fusion = nn.Linear(4 * self.n_embedding, self.n_hidden)

        self.lin1 = nn.Linear(self.n_hidden, int(self.n_hidden / 2))
        self.lin2 = nn.Linear(int(self.n_hidden / 2), 1)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = F.leaky_relu(self.feature_embeding_d(discrete_x))
        x_c1 = F.leaky_relu(self.feature_embeding_c1(continous_x[:, :13]))
        x_c2 = F.leaky_relu(self.feature_embeding_c2(continous_x[:, 13:26]))

        x_g = F.leaky_relu(self.feature_embeding_g0(discrete_x))
        x_g = F.leaky_relu(self.feature_embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.feature_embeding_g2(x_g, edge_index))

        h_ci = torch.cat([x_d, x_c1, x_c2, x_g], dim=1)
        h_ci = F.leaky_relu(self.fusion(h_ci))

        s_ci = F.leaky_relu(self.lin1(h_ci))
        s_ci = F.dropout(s_ci, p=self.dropout, training=self.training)
        s_ci = self.lin2(s_ci)
        s_ci = torch.sigmoid(s_ci).squeeze()

        h_si = h_ci.clone()
        s_si = s_ci.clone()
        y = s_ci

        return (y, s_ci, s_si, h_ci, h_si)


class Base_model(nn.Module):
    def __init__(self, args):
        super(Base_model, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.dropout = args.dropout
        self.device = args.device

        self.feature_embeding_d = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        # self.feature_embeding_c1 = nn.Linear(
        #     self.n_continous_features, self.n_embedding)
        # self.feature_embeding_c2 = nn.Linear(
        #     self.n_continous_features, self.n_embedding)
        # self.feature_embeding_c3 = nn.Linear(
        #     self.n_continous_features, self.n_embedding)
        self.feature_embeding_g0 = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        self.feature_embeding_g1 = GCNConv(self.n_embedding, self.n_embedding)
        self.feature_embeding_g2 = GCNConv(self.n_embedding, self.n_embedding)
        self.fusion = nn.Linear(2 * self.n_embedding, self.n_hidden)

        self.feature_embeding_si_nf0 = nn.Linear(
            self.n_discrete_features, self.n_hidden)
        self.feature_embeding_si_nf1 = GCNConv(self.n_hidden, self.n_hidden)
        self.feature_embeding_si_nf2 = GCNConv(self.n_hidden, self.n_hidden)

        self.feature_embeding_si_ns0 = nn.Linear(
            self.n_churn_state, int(self.n_hidden / 2))
        self.feature_embeding_si_ns1 = GCNConv(int(self.n_hidden / 2), self.n_hidden)

        self.lin1 = nn.Linear(self.n_hidden, int(self.n_hidden / 2))
        self.lin2 = nn.Linear(int(self.n_hidden / 2), 1)
        self.lin3 = nn.Linear(self.n_hidden, int(self.n_hidden / 2))
        self.lin4 = nn.Linear(int(self.n_hidden / 2), 1)

        self.res1 = nn.Linear(2, 2)
        self.res2 = nn.Linear(2, 1)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = F.leaky_relu(self.feature_embeding_d(discrete_x))
        # x_c1 = F.leaky_relu(self.feature_embeding_c1(continous_x[:, :13]))
        # x_c2 = F.leaky_relu(self.feature_embeding_c2(continous_x[:, 13:26]))
        # x_c3 = F.leaky_relu(self.feature_embeding_c3(continous_x[:, 26:39]))

        x_g = F.leaky_relu(self.feature_embeding_g0(discrete_x))
        x_g = F.leaky_relu(self.feature_embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.feature_embeding_g2(x_g, edge_index))

        # h_ci = torch.cat([x_d, x_c1, x_c2, x_c3, x_g], dim=1)
        h_ci = torch.cat([x_d, x_g], dim=1)
        h_ci = F.leaky_relu(self.fusion(h_ci))

        x_si_f = F.leaky_relu(self.feature_embeding_si_nf0(discrete_x))
        x_si_f = F.leaky_relu(self.feature_embeding_si_nf1(x_si_f, edge_index))
        x_si_f = F.leaky_relu(self.feature_embeding_si_nf2(x_si_f, edge_index))

        x_si_ns = F.leaky_relu(self.feature_embeding_si_ns0(churn_date))
        x_si_ns = F.leaky_relu(self.feature_embeding_si_ns1(x_si_ns, edge_index))

        h_si = x_si_f * x_si_ns

        s_ci = F.leaky_relu(self.lin1(h_ci))
        s_ci = F.dropout(s_ci, p=self.dropout, training=self.training)
        s_ci = self.lin2(s_ci)
        s_ci = torch.sigmoid(s_ci)

        s_si = F.leaky_relu(self.lin3(h_si))
        s_si = F.dropout(s_si, p=self.dropout, training=self.training)
        s_si = self.lin4(s_si)
        s_si = torch.sigmoid(s_si)

        y = torch.cat([s_ci, s_si], dim=1)
        y = self.res2(F.leaky_relu(self.res1(y)))
        y = torch.sigmoid(y).squeeze()

        return (y, s_ci, s_si, h_ci, h_si)


class Base_model_c(nn.Module):
    '''
    加上了连续变量，用处似乎不大。。
    '''
    def __init__(self, args):
        super(Base_model_c, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.dropout = args.dropout
        self.device = args.device

        self.feature_embeding_d = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        self.feature_embeding_c1 = nn.Linear(
            self.n_continous_features, self.n_embedding)
        self.feature_embeding_c2 = nn.Linear(
            self.n_continous_features, self.n_embedding)
        self.feature_embeding_c3 = nn.Linear(
            self.n_continous_features, self.n_embedding)
        self.feature_embeding_g0 = nn.Linear(
            self.n_discrete_features, self.n_embedding)
        self.feature_embeding_g1 = GCNConv(self.n_embedding, self.n_embedding)
        self.feature_embeding_g2 = GCNConv(self.n_embedding, self.n_embedding)
        self.fusion = nn.Linear(4 * self.n_embedding, self.n_hidden)

        self.feature_embeding_si_nf0 = nn.Linear(
            self.n_discrete_features, self.n_hidden)
        self.feature_embeding_si_nf1 = GCNConv(self.n_hidden, self.n_hidden)
        self.feature_embeding_si_nf2 = GCNConv(self.n_hidden, self.n_hidden)

        self.feature_embeding_si_ns0 = nn.Linear(
            self.n_churn_state, int(self.n_hidden / 2))
        self.feature_embeding_si_ns1 = GCNConv(int(self.n_hidden / 2), self.n_hidden)

        self.lin1 = nn.Linear(self.n_hidden, int(self.n_hidden / 2))
        self.lin2 = nn.Linear(int(self.n_hidden / 2), 1)
        self.lin3 = nn.Linear(self.n_hidden, int(self.n_hidden / 2))
        self.lin4 = nn.Linear(int(self.n_hidden / 2), 1)

        self.res1 = nn.Linear(2, 2)
        self.res2 = nn.Linear(2, 1)

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = F.leaky_relu(self.feature_embeding_d(discrete_x))
        x_c1 = F.leaky_relu(self.feature_embeding_c1(continous_x[:, :13]))
        x_c2 = F.leaky_relu(self.feature_embeding_c2(continous_x[:, 13:26]))
        # x_c3 = F.leaky_relu(self.feature_embeding_c3(continous_x[:, 26:39]))

        x_g = F.leaky_relu(self.feature_embeding_g0(discrete_x))
        x_g = F.leaky_relu(self.feature_embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.feature_embeding_g2(x_g, edge_index))

        h_ci = torch.cat([x_d, x_c1, x_c2, x_g], dim=1)
        h_ci = F.leaky_relu(self.fusion(h_ci))

        x_si_f = F.leaky_relu(self.feature_embeding_si_nf0(discrete_x))
        x_si_f = F.leaky_relu(self.feature_embeding_si_nf1(x_si_f, edge_index))
        x_si_f = F.leaky_relu(self.feature_embeding_si_nf2(x_si_f, edge_index))

        x_si_ns = F.leaky_relu(self.feature_embeding_si_ns0(churn_date))
        x_si_ns = F.leaky_relu(self.feature_embeding_si_ns1(x_si_ns, edge_index))

        h_si = x_si_f * x_si_ns

        s_ci = F.leaky_relu(self.lin1(h_ci))
        s_ci = F.dropout(s_ci, p=self.dropout, training=self.training)
        s_ci = self.lin2(s_ci)
        s_ci = torch.sigmoid(s_ci)

        s_si = F.leaky_relu(self.lin3(h_si))
        s_si = F.dropout(s_si, p=self.dropout, training=self.training)
        s_si = self.lin4(s_si)
        s_si = torch.sigmoid(s_si)

        y = torch.cat([s_ci, s_si], dim=1)
        y = self.res2(F.leaky_relu(self.res1(y)))
        y = torch.sigmoid(y).squeeze()

        return (y, s_ci, s_si, h_ci, h_si)


class ResDCN_dual_attn(nn.Module):
    '''
    '''
    def __init__(self, args):
        super(ResDCN_dual_attn, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = 4
        self.n_channels_d = 6

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden

        # self intention
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             'leaky_relu', use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = nn.Linear(2 * self.nh1, self.nh1)
        self.prediction_ci = PredictionLayer(self.nh1, [int(self.nh1/2)], 'leaky_relu')

        # social influence
        self.embeding_si_nf0 = nn.Linear(self.nh1, self.nh1)
        self.embeding_si_nf1 = EGATConv(self.nh1, self.nh1, self.n_edge_features, heads=1)
        self.embeding_si_nf2 = EGATConv(self.nh1, self.nh1, self.n_edge_features, heads=1)

        self.attn_ci = nn.Linear(2*self.nh1, self.nh1, bias=False)
        self.attn_si = nn.Linear(2*self.nh1, self.nh1, bias=False)

        # self.embeding_si_ns0 = nn.Linear(
        #     self.n_churn_state, int(self.n_hidden / 2))
        # self.embeding_si_ns1 = GCNConv(int(self.n_hidden / 2), self.n_hidden)
        # self.fusion_si = nn.Linear(self.n_hidden, self.n_embedding)
        self.prediction_si = PredictionLayer(self.nh1, [int(self.nh1/2)], 'leaky_relu')

        self.prediction_y = PredictionLayer(2, [2], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = F.leaky_relu(self.embeding_g0(x_g))
        x_g = F.leaky_relu(self.embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        # self intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = torch.cat([x_deep, x_cross], dim=-1)
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = F.leaky_relu(self.fusion_ci(h_ci)) + x

        # social influence
        x_si_f = F.leaky_relu(self.embeding_si_nf0(x))
        x_si_f = F.leaky_relu(self.embeding_si_nf1(x_si_f, edge_index, edge_attr))
        x_si_f = F.leaky_relu(self.embeding_si_nf2(x_si_f, edge_index, edge_attr))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns0(churn_date))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns1(x_si_ns, edge_index))

        # h_si = x_si_f * x_si_ns
        h_si = x_si_f
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)

        # prediction
        h_ci_si = torch.cat([h_ci, h_si], dim=-1)
        alpha_ci = self.attn_ci(h_ci_si)
        alpha_ci = torch.softmax(alpha_ci, dim=1)
        s_ci = self.prediction_ci(h_ci * alpha_ci)

        alpha_si = self.attn_si(h_ci_si)
        alpha_si = torch.softmax(alpha_si, dim=1)
        s_si = self.prediction_si(h_si * alpha_si)

        pred_y = torch.cat([s_ci, s_si], dim=1)
        pred_y = self.prediction_y(pred_y)

        return (pred_y, s_ci, s_si, h_ci, h_si)


class ResDCN_dual_attn2(nn.Module):
    '''
    在dual_attn的基础上加入了edge_feature_embedding;
    同时，把对边的表征从只让edge_feature加入attn_weights的计算，改进为，加入把edge_feature同时作为attn的value
    '''
    def __init__(self, args):
        super(ResDCN_dual_attn2, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = 4
        self.n_channels_d = 6

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden

        # embeddng
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)
        self.embedding_e = DNN(self.n_edge_features, [self.n_edge_features]*2, use_bn=True)

        # churn intention
        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             'leaky_relu', use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = nn.Linear(2 * self.nh1, self.nh1)
        self.prediction_ci = PredictionLayer(self.nh1, [int(self.nh1/2)], 'leaky_relu')

        # social influence
        self.embeding_si_nf0 = nn.Linear(self.nh1, self.nh1)
        self.embeding_si_nf1 = EGATConv(self.nh1, self.nh1, self.n_edge_features, heads=1)
        self.embeding_si_nf2 = EGATConv(self.nh1, self.nh1, self.n_edge_features, heads=1)

        self.attn_ci = nn.Linear(2*self.nh1, self.nh1, bias=False)
        self.attn_si = nn.Linear(2*self.nh1, self.nh1, bias=False)

        # self.embeding_si_ns0 = nn.Linear(
        #     self.n_churn_state, int(self.n_hidden / 2))
        # self.embeding_si_ns1 = GCNConv(int(self.n_hidden / 2), self.n_hidden)
        # self.fusion_si = nn.Linear(self.n_hidden, self.n_embedding)
        self.prediction_si = PredictionLayer(self.nh1, [int(self.nh1/2)], 'leaky_relu')

        self.prediction_y = PredictionLayer(2, [2], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        edge_attr = self.embedding_e(edge_attr)

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = F.leaky_relu(self.embeding_g0(x_g))
        x_g = F.leaky_relu(self.embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        # self intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = torch.cat([x_deep, x_cross], dim=-1)
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = F.leaky_relu(self.fusion_ci(h_ci)) + x

        # social influence
        x_si_f = F.leaky_relu(self.embeding_si_nf0(x))
        x_si_f = F.leaky_relu(self.embeding_si_nf1(x_si_f, edge_index, edge_attr))
        x_si_f = F.leaky_relu(self.embeding_si_nf2(x_si_f, edge_index, edge_attr))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns0(churn_date))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns1(x_si_ns, edge_index))

        # h_si = x_si_f * x_si_ns
        h_si = x_si_f
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)

        # prediction
        h_ci_si = torch.cat([h_ci, h_si], dim=-1)
        alpha_ci = self.attn_ci(h_ci_si)
        alpha_ci = torch.softmax(alpha_ci, dim=1)
        s_ci = self.prediction_ci(h_ci * alpha_ci)

        alpha_si = self.attn_si(h_ci_si)
        alpha_si = torch.softmax(alpha_si, dim=1)
        s_si = self.prediction_si(h_si * alpha_si)

        pred_y = torch.cat([s_ci, s_si], dim=1)
        pred_y = self.prediction_y(pred_y)

        return (pred_y, s_ci, s_si, h_ci, h_si)


class ResDCN_dual_attn3(nn.Module):
    '''
    在dual_attn的基础上加入了edge_feature_embedding;
    同时，把对边的表征从只让edge_feature加入attn_weights的计算，改进为，加入把edge_feature同时作为attn的value
    '''
    def __init__(self, args):
        super(ResDCN_dual_attn3, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden
        self.n_edge_features = args.n_edge_features
        self.n_channels_c = 4
        self.n_channels_d = 6

        self.nh0 = self.n_discrete_features - 6 + 3 * self.n_channels_c
        self.nh1 = self.nh0 + self.n_hidden

        # self intention
        self.embedding_d = DiscreteFeatureEmbeddingLayer()
        self.embedding_c = ContinousFeatureEmbeddingLayer(
            self.n_continous_features, self.n_channels_c)
        self.embeding_g0 = nn.Linear(self.nh0, self.n_hidden)
        self.embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)
        self.embedding_e = DNN(self.n_edge_features, [self.n_edge_features]*2, use_bn=True)

        self.resdnn = ResDNN(self.nh1, self.l_hidden,
                             'leaky_relu', use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh1, 2)
        self.fusion_ci = nn.Linear(2 * self.nh1, self.nh1)
        self.prediction_ci = PredictionLayer(self.nh1, [int(self.nh1/2)], 'leaky_relu')

        # social influence
        self.embeding_si_nf0 = nn.Linear(self.nh1, self.n_hidden)
        self.embeding_si_nf1 = EGATConv(
            self.n_hidden, self.n_hidden, self.n_edge_features, heads=2, concat=False)
        self.embeding_si_nf2 = EGATConv(
            self.n_hidden, self.n_hidden, self.n_edge_features, heads=2, concat=False)

        self.attn_ci = nn.Linear(self.nh1 + self.n_hidden, self.nh1, bias=False)
        self.attn_si = nn.Linear(self.nh1 + self.n_hidden, self.n_hidden, bias=False)

        # self.embeding_si_ns0 = nn.Linear(
        #     self.n_churn_state, int(self.n_hidden / 2))
        # self.embeding_si_ns1 = GCNConv(int(self.n_hidden / 2), self.n_hidden)
        # self.fusion_si = nn.Linear(self.n_hidden, self.n_embedding)
        self.prediction_si = PredictionLayer(self.n_hidden, [int(self.n_hidden/2)], 'leaky_relu')

        self.prediction_y = PredictionLayer(2, [2], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date

        edge_attr = self.embedding_e(edge_attr)

        x_d = self.embedding_d(discrete_x)
        x_c = self.embedding_c(continous_x)
        x_g = torch.cat([x_d, x_c], dim=-1)
        x_g = F.leaky_relu(self.embeding_g0(x_g))
        x_g = F.leaky_relu(self.embeding_g1(x_g, edge_index))
        x_g = F.leaky_relu(self.embeding_g2(x_g, edge_index))
        x = torch.cat([x_d, x_c, x_g], dim=-1)

        # self intention
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = torch.cat([x_deep, x_cross], dim=-1)
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = F.leaky_relu(self.fusion_ci(h_ci)) + x

        # social influence
        x_si_f = F.leaky_relu(self.embeding_si_nf0(x))
        x_si_f = F.leaky_relu(self.embeding_si_nf1(x_si_f, edge_index, edge_attr))
        x_si_f = F.leaky_relu(self.embeding_si_nf2(x_si_f, edge_index, edge_attr))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns0(churn_date))
        # x_si_ns = F.leaky_relu(self.embeding_si_ns1(x_si_ns, edge_index))

        # h_si = x_si_f * x_si_ns
        h_si = x_si_f
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)

        # prediction
        h_ci_si = torch.cat([h_ci, h_si], dim=-1)
        alpha_ci = self.attn_ci(h_ci_si)
        alpha_ci = torch.softmax(alpha_ci, dim=1)
        s_ci = self.prediction_ci(h_ci * alpha_ci)

        alpha_si = self.attn_si(h_ci_si)
        alpha_si = torch.softmax(alpha_si, dim=1)
        s_si = self.prediction_si(h_si * alpha_si)

        pred_y = torch.cat([s_ci, s_si], dim=1)
        pred_y = self.prediction_y(pred_y)

        return (pred_y, s_ci, s_si, h_ci, h_si)


class GCNEncoder(nn.Module):
    """docstring for GCNEncoder"""
    def __init__(self, n_input, n_output, activation='leaky_relu'):
        super(GCNEncoder, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.activates = activation_layer(activation)
        self.layer1 = GCNConv(self.n_input, self.n_output)
        self.layer2 = GCNConv(self.n_output, self.n_output)

    def forward(self, x, edge_index):
        z = self.activates(self.layer1(x, edge_index))
        z = self.layer2(z, edge_index)
        return z


class InnerProductDecoder(nn.Module):
    r"""
    The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper
    """
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`, shape n*dz.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class GAE(nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.  InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True)
                              + EPS).mean()
        # # Do not include self-loops in negative samples
        # pos_edge_index, _ = remove_self_loops(pos_edge_index)
        # pos_edge_index, _ = add_self_loops(pos_edge_index)
        # if neg_edge_index is None:
        #     neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class CFChurn_GAE(nn.Module):
    """docstring for CounterfactualChurn"""
    def __init__(self, args):
        super(CFChurn_GAE, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden

        self.nh = self.n_discrete_features - 6
        self.act = activation_layer(args.activation)

        # churn intention module
        self.feature_embedding_d = DiscreteFeatureEmbeddingLayer()
        self.feature_embeding_g0 = nn.Linear(self.nh, self.n_hidden)
        self.feature_embeding_g1 = GCNConv(self.n_hidden, self.n_hidden)
        self.feature_embeding_g2 = GCNConv(self.n_hidden, self.n_hidden)

        self.resdnn = ResDNN(self.nh + self.n_hidden, self.l_hidden,
                             'leaky_relu', use_bn=self.bn,
                             dropout_rate=self.dropout)
        self.cross = CrossNet(self.nh + self.n_hidden, 2)
        self.fusion_ci = nn.Linear(2 * (self.nh + self.n_hidden), self.n_embedding)
        self.prediction_ci = PredictionLayer(self.n_embedding, [int(self.n_embedding/2)], 'leaky_relu')

        # social influence module
        self.feature_embedding_d_si = DiscreteFeatureEmbeddingLayer()
        self.resdnn_si = ResDNN(self.nh, self.l_hidden, 'leaky_relu',
                                use_bn=self.bn, dropout_rate=self.dropout)
        self.cross_si = CrossNet(self.nh, 2)
        self.fusion_si = nn.Linear(2 * self.nh, self.n_embedding)

        self.encoder = GCNEncoder(self.nh, self.nh)
        self.decoder = InnerProductDecoder()
        self.gae = GAE(self.encoder, self.decoder)
        self.lin1_si = nn.Linear(self.nh, self.n_embedding)

        self.y0_predicter = PredictionLayer(self.n_embedding, [self.n_embedding] * 2, 'leaky_relu')
        self.y1_predicter = PredictionLayer(self.n_embedding, [self.n_embedding] * 2, 'leaky_relu')
        self.T_predicter = PredictionLayer(self.n_embedding, [], 'leaky_relu')
        self.y_predicter = PredictionLayer(2, [2], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t, neg_edge_index = \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t, data.neg_edge_index

        # churn intention
        x_d = self.feature_embedding_d(discrete_x)
        x_g = self.act(self.feature_embeding_g0(x_d))
        x_g = self.act(self.feature_embeding_g1(x_g, edge_index))
        x_g = self.act(self.feature_embeding_g2(x_g, edge_index))

        x = torch.cat([x_d, x_g], dim=-1)
        x_deep = self.resdnn(x)
        x_cross = self.cross(x)
        h_ci = torch.cat([x_deep, x_cross], dim=-1)
        h_ci = F.dropout(h_ci, p=self.dropout, training=self.training)
        h_ci = self.act(self.fusion_ci(h_ci))
        y_ci = self.prediction_ci(h_ci)

        # social influence
        x_d_si = self.feature_embedding_d_si(discrete_x)
        x_deep_si = self.resdnn_si(x_d_si)
        x_cross_si = self.cross_si(x_d_si)
        x_self = torch.cat([x_deep_si, x_cross_si], dim=-1)
        x_self = F.dropout(x_self, p=self.dropout, training=self.training)
        x_self = self.fusion_si(x_self)

        z = self.gae.encode(x_d_si, edge_index)
        z = F.dropout(z, p=self.dropout, training=self.training)
        x_neighbors = self.act(self.lin1_si(z))

        h_si = x_self * x_neighbors

        pred_T = self.T_predicter(h_si)
        pred_y0 = self.y0_predicter(h_si)
        pred_y1 = self.y1_predicter(h_si)

        pred_T = pred_T[:len(t), :]
        pred_y0 = pred_y0[:len(t), :]
        pred_y1 = pred_y1[:len(t), :]
        y_ci = y_ci[:len(t), :]

        y_f = (1 - t) * pred_y0 + t * pred_y1
        y_cf = t * pred_y0 + (1 - t) * pred_y1
        # y = torch.max(torch.cat([y_ci, y_f], dim=-1), dim=-1)[0].view(-1, 1)
        y = self.y_predicter(torch.cat([y_ci, y_f], dim=-1))

        gae_loss = self.gae.recon_loss(z, edge_index, neg_edge_index)

        return (y, y_ci, y_cf, pred_T, h_ci, h_si, gae_loss)


class CFChurn_si_only(nn.Module):
    """docstring for CounterfactualChurn"""
    def __init__(self, args):
        super(CFChurn_si_only, self).__init__()
        self.args = args
        self.n_discrete_features = args.n_discrete_features
        self.n_continous_features = int(args.n_continous_features / 3)
        self.n_churn_state = args.n_churn_state
        self.n_hidden = args.n_hidden
        self.n_embedding = args.n_embedding
        self.bn = bool(args.batch_norm)
        self.dropout = args.dropout
        self.l_hidden = args.l_hidden

        self.nh = self.n_discrete_features - 6
        self.act = activation_layer(args.activation)

        # social influence module
        self.feature_embedding_d_si = DiscreteFeatureEmbeddingLayer()
        self.feature_embeding_g0_si = nn.Linear(self.nh, self.nh)
        self.feature_embeding_g1_si = GCNConv(self.nh, self.nh)
        self.feature_embeding_g2_si = GCNConv(self.nh, self.nh)

        self.resdnn_si = ResDNN(2*self.nh, self.l_hidden, 'leaky_relu',
                                use_bn=self.bn, dropout_rate=self.dropout)
        self.cross_si = CrossNet(2*self.nh, 2)
        self.fusion_si_1 = nn.Linear(4*self.nh, 2*self.nh)
        self.fusion_si_2 = nn.Linear(2*self.nh, self.n_embedding)

        self.y0_predicter = PredictionLayer(self.n_embedding, [self.n_embedding] * 2, 'leaky_relu')
        self.y1_predicter = PredictionLayer(self.n_embedding, [self.n_embedding] * 2, 'leaky_relu')
        self.T_predicter = PredictionLayer(self.n_embedding, [], 'leaky_relu')

    def forward(self, data):
        discrete_x, continous_x, edge_index, edge_attr, churn_date, t= \
            data.discrete_x, data.continous_x, data.edge_index, \
            data.edge_attr, data.churn_date, data.t

        # social influence
        x_d_si = self.feature_embedding_d_si(discrete_x)
        x_g_si = self.act(self.feature_embeding_g0_si(x_d_si))
        x_g_si = self.act(self.feature_embeding_g1_si(x_g_si, edge_index))
        x_g_si = self.act(self.feature_embeding_g2_si(x_g_si, edge_index))
        x_si = torch.cat([x_d_si, x_g_si], dim=-1)

        x_deep_si = self.resdnn_si(x_si)
        x_cross_si = self.cross_si(x_si)
        h_si = torch.cat([x_deep_si, x_cross_si], dim=-1)

        h_si = self.act(self.fusion_si_1(h_si))
        h_si = F.dropout(h_si, p=self.dropout, training=self.training)
        h_si = self.act(self.fusion_si_2(h_si))

        pred_T = self.T_predicter(h_si)
        pred_y0 = self.y0_predicter(h_si)
        pred_y1 = self.y1_predicter(h_si)

        pred_T = pred_T[:len(t), :]
        pred_y0 = pred_y0[:len(t), :]
        pred_y1 = pred_y1[:len(t), :]

        y_f = (1 - t) * pred_y0 + t * pred_y1
        y_cf = t * pred_y0 + (1 - t) * pred_y1

        return (y_f, y_cf, pred_T, h_si)
