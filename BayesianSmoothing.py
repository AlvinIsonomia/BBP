#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from utils.train_parse_args import parse_args
from utils.dataloader import load_local_data
import random
import scipy.special as special
from utils.s3utils import S3FileSystemPatched
import s3fs
import pandas as pd
from utils.s3utils import S3Filewrite, S3FileSystemPatched, S3Filewrite_bayes


def save_to_s3(src_path, dst_path):
    cmd = 's3cmd put -r ' + src_path + ' ' + dst_path
    os.system(cmd)

def load_s3_data(path):

    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()

    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    print(input_files[:3])

    index = 0
    for file in input_files:
        if index == 0:
            feature = pd.read_csv("s3://" + file, header=None).values

        if index > 0:
            feature_ = pd.read_csv("s3://" + file, header=None).values
            feature = np.r_[feature, feature_]
        index += 1
        print(index)
    feature = np.array(feature)
    return feature

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.00000001
        numerator_beta = 0.00000001
        denominator = 0.00000001

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))
#         print(denominator)
        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)


def train(args):

    # 1. load and split data 

    file_path = args.data_input.split(',')[0]
    base_p_path = args.data_input.split(',')[1]

    feature = load_s3_data(file_path)
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_file = sorted([file for file in fs.ls(base_p_path) if file.find("part-") != -1])
    [base_alpha, base_beta] = list(map(float, pd.read_csv("s3://" + input_file[0], header=None).values[0]))
    print("data_loading done!")

    id, I, C = feature[:, 0], feature[:, 1], feature[:, 2]
    pre = []
    alpha = []
    beta = []
    nodeid = []
    for i in range(len(I)):
        nodeid.append(id[i])
        x, y = list(map(float, str(I[i]).split(" "))), list(map(float, str(C[i]).split(" ")))
        bs = BayesianSmoothing(base_alpha, base_beta)
        bs.update(x, y, args.iter_num, args.epsilon)
        ctr = []
        for j in range(len(x)):
            ctr.append((y[j]+bs.alpha)/(x[j]+bs.alpha+bs.beta))
        pre.append(ctr[-1])
        alpha.append(bs.alpha)
        beta.append(bs.beta)
        if i%10000 == 0:
            print("pross:",i)
    batch_size = 60000
    for idx in range(0, len(id), batch_size):
        s3writer = S3Filewrite_bayes(args)
        s3writer.write(nodeid[idx:idx + batch_size], pre[idx:idx + batch_size], alpha[idx:idx + batch_size], beta[idx:idx + batch_size], idx)
        print("write_batch_60000:", idx)


def train_local(args):

    # 1. load and split data 

    # file_path = args.data_input.split(',')[0]
    # base_p_path = args.data_input.split(',')[1]

    # feature = load_s3_data(file_path)
    # s3fs.S3FileSystem = S3FileSystemPatched
    # fs = s3fs.S3FileSystem()
    # input_file = sorted([file for file in fs.ls(base_p_path) if file.find("part-") != -1])

    file_path = args.file_path
    feature = pd.read_csv(file_path, header=0).values
    # alo
    base_alpha = args.alpha
    base_beta = args.beta
    # [base_alpha, base_beta] = list(map(float, pd.read_csv("s3://" + input_file[0], header=None).values[0]))
    
    print("data_loading done!")

    id, I, C = feature[:, 0], feature[:, 1], feature[:, 2]
    pre = []
    alpha = []
    beta = []
    nodeid = []
    for i in range(len(I)):
        nodeid.append(id[i])
        x, y = list(map(float, str(I[i]).split(" "))), list(map(float, str(C[i]).split(" ")))
        bs = BayesianSmoothing(base_alpha, base_beta)
        bs.update(x, y, args.iter_num, args.epsilon)
        ctr = []
        for j in range(len(x)):
            ctr.append((y[j]+bs.alpha)/(x[j]+bs.alpha+bs.beta))
        pre.append(ctr[-1])
        alpha.append(bs.alpha)
        beta.append(bs.beta)
        if i%10000 == 0:
            print("pross:",i)
    batch_size = 60000
    # save to csv file
    df = pd.DataFrame({'nodeid': nodeid, 'pre': pre, 'alpha': alpha, 'beta': beta})
    df.to_csv(args.save_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    # train(args)
    train_local(args)
