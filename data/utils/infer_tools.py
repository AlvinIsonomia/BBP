# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from inference import sparkfuel as sf
from config.config import CODM


class it_wrapper(object):
    def __init__(self, it):
        self.it = it
        self._hasnext = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.it)
        self._hasnext = False
        return result

    def hasnext(self):
        if self._hasnext is False:
            try:
                self._thenext = next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext


# run linux commands
def run_cmd(args_list):
    """
    usage:
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-ls', 'hdfs_file_path'])
    lines = out.split('\n')
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-put', 'local_file', 'hdfs_file_path'])
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-rm', 'hdfs_file_path'])
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-rm', '-skipTrash', 'hdfs_file_path'])
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-rm', '-r', 'hdfs_file_path'])
    (ret, out, err)= run_cmd(['hdfs', 'dfs', '-rm', '-r', '-skipTrash', 'hdfs_file_path'])

    # Check if a file exist in HDFS
    # hadoop fs -test -e filename
    hdfs_file_path = '/tmpo'
    cmd = ['hdfs', 'dfs', '-test', '-e', hdfs_file_path]
    ret, out, err = run_cmd(cmd)
    print(ret, out, err)
    if ret:
        print('file does not exist')
    """
    from subprocess import PIPE, Popen
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = Popen(args_list, stdout=PIPE, stderr=PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err



# Load item feature from given directory
def load_item_data(inp_hdfs_dir, local_save_dir):
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    #
    hadoop_home = os.path.join(os.getcwd(), "tdwgaia")
    if not os.path.exists(hadoop_home):
        if 'HADOOP_PREFIX' in os.environ:
            hadoop_home = os.environ['HADOOP_PREFIX']
        elif 'HADOOP_HOME' in os.environ:
            hadoop_home = os.environ['HADOOP_HOME']
        else:
            raise RuntimeError("hadoop home not found")
    ugi = os.environ.get('UGI')
    ugi_part = "-Dhadoop.job.ugi={}".format(ugi) if ugi else ""
    cmd_array = ["{}/bin/hadoop".format(hadoop_home), "fs", "-get", inp_hdfs_dir + "*", local_save_dir]
    print(cmd_array)
    (ret, out, err) = run_cmd(cmd_array)
    print("{} {} {}".format(ret, out, err))

    print(os.listdir(local_save_dir))
    inp_file_paths = list(
        map(lambda x: local_save_dir + x, filter(lambda s: (not s == "_SUCCESS"), os.listdir(local_save_dir))))
    print(inp_file_paths)
    item_df = pd.concat([pd.read_csv(f) for f in inp_file_paths], axis=0, ignore_index=True)
    return item_df



def map_concat_feat_with_key():
    def map_row(row):
        return (tf.cast(row[0], tf.string), tf.cast(row[1], tf.string), tf.cast(row[2], tf.float32))

    return map_row


def trans_func_wrapper_HP(path, model_version):
    if path == "local":
        scaler = pd.read_csv("./model/scaler.csv", header=None).values
    else:
        sf.setup_hadoop_classpath()
        scaler_path = os.path.join(path, 'scaler.csv')
        load_item_data(scaler_path, "./")
        print("load scaler")
        scaler = pd.read_csv('scaler.csv', header=None).values
    fea_idx = CODM.imp_fea_index + CODM.oth_fea_index
    u_max = np.array(scaler[0,])
    u_min = np.array(scaler[1,])

    def _trans_func(row):
        """
        Preprocess features (min-max scale features, replace Inf and Nan values by 1)
        :param row: User feature data.
        :return: Processed user feature data.
        """

        src = row[0]
        dst = row[1]
        feature = np.array(row[2:], dtype=np.float32)
        if model_version == "v2":
            feature = feature[fea_idx]
        feature = (feature - u_min) / (u_max - u_min + 1e-7)
        feature[np.where(np.isnan(feature))] = 0.0
        feature[np.where(np.isinf(feature))] = 1.0
        feature = np.reshape(feature, (1, -1))
        return (src, dst, feature)

    return _trans_func


def get_item_df(item_feat_path, local_dir):
    """
    item_feat_path could be local or remote. If from hdfs, use load_item_data function to load item data.
    Otherwise, load from local dir.
    :param item_feat_path: Could be local or remote.
                           Remote example: hdfs://ss-ieg-dm-v2/data/turing/etanyang/csv_data/skin_feature_with_embedding.csv
                           Local example: /skin_feature_with_embedding_512.csv
    :return: Item feature dataframe.
    """
    if item_feat_path[0:4] == "hdfs":
        item_df = load_item_data(item_feat_path, local_dir)
    else:
        item_df = pd.read_csv(os.getcwd() + item_feat_path, sep=';', header=None)
    return item_df