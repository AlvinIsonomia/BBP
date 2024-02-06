# -*- coding: utf-8 -*-

import os
from utils.platform import run_cmd


def get_hadoop_home():
    
    hadoop_home = os.path.join(os.getcwd(), "tdwgaia")
    if not os.path.exists(hadoop_home):
        if 'HADOOP_PREFIX' in os.environ:
            hadoop_home = os.environ['HADOOP_PREFIX']
        elif 'HADOOP_HOME' in os.environ:
            hadoop_home = os.environ['HADOOP_HOME']
        else:
            hadoop_home = None
    
    return hadoop_home


def get_hadoop_ugi_config():
    ugi = os.environ.get('UGI')
    ugi_part = "-Dhadoop.job.ugi={}".format(ugi) if ugi else ""
    
    return ugi_part


def copy_dir(inp_hdfs_dir, local_save_dir):
    
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    
    hadoop_home = get_hadoop_home()
    if hadoop_home is None:
        raise RuntimeError("hadoop home not found")

    cmd_array = ["{}/bin/hadoop".format(hadoop_home), "fs", "-get", inp_hdfs_dir + "*", local_save_dir]

    (ret, _, err) = run_cmd(cmd_array)
    if ret:
        raise RuntimeError("failed run_cmd [{}] return {}, errmsg {}".format(" ".join(cmd_array), ret, err))
