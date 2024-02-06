# -*- coding: utf-8 -*-
import s3fs
import numpy as np
import pandas as pd

from utils.s3utils import S3FileSystemPatched


def load_s3_data(path, num_fea_size):
    """load data from S3 storage

    Arguments:
        path {string} -- s3 path
        num_fea_size {int} --  number of feature dimension

    Returns:
        Tuple of Array
    """

    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()

    """get file list"""
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    print(input_files[:3])

    """load data from s3 files"""
    index = 0
    for file in input_files:
        if index == 0:
            src = pd.read_csv("s3://" + file, header=None, usecols=[0]).values
            dst = pd.read_csv("s3://" + file, header=None, usecols=[1]).values
            label = pd.read_csv("s3://" + file, header=None, usecols=[2]).values
            feature = pd.read_csv("s3://" + file, header=None, usecols=[i + 3 for i in range(num_fea_size)]).values

        if index > 0:
            src_ = pd.read_csv("s3://" + file, header=None, usecols=[0]).values
            dst_ = pd.read_csv("s3://" + file, header=None, usecols=[1]).values
            label_ = pd.read_csv("s3://" + file, header=None, usecols=[2]).values
            feature_ = pd.read_csv("s3://" + file, header=None, usecols=[i + 3 for i in range(num_fea_size)]).values
            src = np.r_[src, src_]
            dst = np.r_[dst, dst_]
            label = np.r_[label, label_]
            feature = np.r_[feature, feature_]
        index += 1
        print(index)
    label = np.array(label)
    feature = np.array(feature)

    """remove rows which contain INF and NAN"""
    print("INF:", np.isinf(feature).any())
    if np.isinf(feature).any():
        try:
            dst = np.delete(dst, np.where(np.isinf(feature))[0], axis=0)
            src = np.delete(src, np.where(np.isinf(feature))[0], axis=0)
            label = np.delete(label, np.where(np.isinf(feature))[0], axis=0)
            feature = np.delete(feature, np.where(np.isinf(feature))[0], axis=0)
            print("delete inf done")
        except:
            print("delete inf error")
    print("NULL:", np.isnan(feature).any())
    if np.isnan(feature).any():
        try:
            dst = np.delete(dst, np.where(np.isnan(feature))[0], axis=0)
            src = np.delete(src, np.where(np.isnan(feature))[0], axis=0)
            label = np.delete(label, np.where(np.isnan(feature))[0], axis=0)
            feature = np.delete(feature, np.where(np.isnan(feature))[0], axis=0)
            print("delete nan done")
        except:
            print("delete null error")
    return feature, label, src, dst








