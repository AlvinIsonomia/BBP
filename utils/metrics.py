# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def AUC(y_true, y_pred):
    try:
        return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    except:
        return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_offline_apply_user_v2(rec_dict, apply_dict, max_rank=10):
    num_user_full = 0.0
    hitrate_full = 0.0
    ndcg_full = 0.0

    rec_user = list(rec_dict.keys())
    apply_user = list(apply_dict.keys())

    print("start offline evaluation!")

    for i, playerid in enumerate(apply_user):

        rank = 0
        if playerid not in rec_user:
            continue

        rec_list = rec_dict[playerid][:max_rank]
        apply_list = apply_dict[playerid]
        try:
            # while rank < 100 and rec_dict[playerid][rank] not in apply_dict[playerid]:
            for apply_dst in apply_list:
                if apply_dst in rec_list:
                    rank = rec_list.index(apply_dst)
                    ndcg_full += 1.0 / np.log2(rank + 2.0)
                    hitrate_full += 1.0
                num_user_full += 1
        except:
            # while rank < 100 and rec_dict[playerid][rank] != apply_dict[playerid]:
            if rec_list[rank] == apply_list:
                ndcg_full += 1.0 / np.log2(2.0)
                hitrate_full += 1.0
            num_user_full += 1

    ndcg_full /= num_user_full
    hitrate_full /= num_user_full
    return np.array([ndcg_full, hitrate_full], dtype=np.float32)


def evaluate_offline_apply_user(rec_dict, apply_dict, max_rank=100):
    num_user_full = 0.0
    hitrate_full = 0.0
    ndcg_full = 0.0

    rec_user = list(rec_dict.keys())
    apply_user = list(apply_dict.keys())

    print("start offline evaluation!")

    for i, playerid in enumerate(apply_user):
        num_user_full += 1
        rank = 0
        if playerid not in rec_user:
            continue
        # else:
        rec_list = rec_dict[playerid]
        apply_list = apply_dict[playerid]
        try:
            # while rank < 100 and rec_dict[playerid][rank] not in apply_dict[playerid]:
            while rank < max_rank and rec_list[rank] not in apply_list:
                rank += 1
        except:
            # while rank < 100 and rec_dict[playerid][rank] != apply_dict[playerid]:
            while rank < max_rank and rec_list[rank] != apply_list:
                rank += 1

        if rank < max_rank:
            ndcg_full += 1.0 / np.log2(rank + 2.0)
            hitrate_full += 1.0

    ndcg_full /= num_user_full
    hitrate_full /= num_user_full

    return np.array([ndcg_full, hitrate_full], dtype=np.float32)