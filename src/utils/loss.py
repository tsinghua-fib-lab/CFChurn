import torch
import torch.nn.functional as F


def loss_func(args, ids, pred_y, y, t, s_si, h_ci, h_si):
    return ((F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
             torch.abs(s_si[ids, :]) *
             (args.causal_weight1 * (t[ids, :] == 0).float() * y[ids, :] +
              args.causal_weight2 * t[ids, :] * (y[ids, :] == 0).float())) *
            (1 + args.imbalance_penalty * y[ids, :]))
# args.disentanglement_weight * torch.abs(torch.cosine_similarity(h_si[ids], h_ci[ids], dim=1)) + \


def loss_func2(args, ids, pred_y, y):
    '''supervised loss only'''
    return F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') * \
        (1 + args.imbalance_penalty * y[ids, :])


def loss_func3(args, ids, pred_y, y, pred_y_cf, y_cf, id_cf, pred_T, t):
    return ((F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
            F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :]) *
            (1 + args.imbalance_penalty * y[ids, :]) +
            F.binary_cross_entropy(pred_T[ids, :], t[ids, :], reduction='none') * args.treatment_weight)


def loss_func4(args, ids, pred_y, y, pred_y_cf, y_cf, id_cf, pred_T, t):
    '''对不同的treatment支路加了不同的imbalance penalty'''
    return ((F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
            F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :]) *
            (1 + args.imbalance_penalty_t1 * y[ids, :]) * t[ids, :] +
            (F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
            F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :]) *
            (1 + args.imbalance_penalty_t0 * y[ids, :]) * (1 - t[ids, :]) +
            F.binary_cross_entropy(pred_T[ids, :], t[ids, :], reduction='none') * args.treatment_weight)


def loss_func5(args, ids, pred_y, y, pred_y_cf, y_cf, id_cf, pred_T, t, pred_y0, pred_y1):
    '''对不同的treatment支路加了不同的imbalance penalty, 同时，给不同的cf加上不同的loss，看看哪种cf有用'''
    return ((1 + args.imbalance_penalty_t1 * y[ids, :]) * t[ids, :] *
            (F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
             F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :] * args.cf_weight1) +

            (1 + args.imbalance_penalty_t0 * y[ids, :]) * (1 - t[ids, :]) *
            (F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
             F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :] * args.cf_weight2) +

            F.binary_cross_entropy(pred_T[ids, :], t[ids, :], reduction='none') * args.treatment_weight +

            torch.max(torch.cat([pred_y0[ids, :] - pred_y1[ids, :], torch.zeros(pred_y0[ids, :].size(0), 1).to(args.device)], dim=-1), dim=-1).values.view(-1, 1) * args.causal_weight * id_cf[ids, :])
