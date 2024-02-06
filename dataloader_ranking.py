import numpy as np
import pandas as pd
import sklearn.preprocessing
import os


def get_pointwise_set(data):
    '''
    :param data: the training, validation or testing data with pointwise format (pd.DataFrame)
    :return: the user id list, item id list, label list and feature matrix
    '''
    user_id_list = data['userid'].to_numpy()
    item_id_list = data['itemid'].to_numpy()
    label_list = data['label'].to_numpy()
    feature_matrix = data.iloc[:, 3:].to_numpy()
    return user_id_list, item_id_list, label_list, feature_matrix

def get_bbp_set(data):
    '''
    :param data: the training, validation or testing data with pointwise format (pd.DataFrame)
    :return: the user id list, item id list, label list and feature matrix
    '''
    user_id_list = data['userid'].to_numpy()
    item_id_list = data['itemid'].to_numpy()
    label_list = data['label'].to_numpy()
    score_list = data['score'].to_numpy()
    feature_matrix = data.iloc[:, 4:].to_numpy()
    return user_id_list, item_id_list, label_list, score_list, feature_matrix