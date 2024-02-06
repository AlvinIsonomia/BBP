import argparse

import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
from utils.dataloader import load_s3_data
from utils.s3utils import S3Filewrite, S3FileSystemPatched

def save_to_s3(src_path, dst_path):
    cmd = 's3cmd put -r ' + src_path + ' ' + dst_path
    os.system(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', default='')
    parser.add_argument(
        "--data_input",
        type=str,
    )
    parser.add_argument(
        "--data_output",
        type=str,
    )
    parser.add_argument(
        "--tb_log_dir",
        type=str,
    )

    args = parser.parse_args()

    file_path = args.data_input.split(',')[0]
    feature, label, src, dst = load_s3_data(file_path, num_fea_size=768)
    pca = PCA(n_components=128)
    pca_result = pca.fit_transform(feature)
    print("pca done!")

    final_array = np.c_[src, dst, label, pca_result]

    os.makedirs("emb", exist_ok=True)
    np.savetxt("emb/pca_emb.csv", final_array, delimiter=',', fmt="%s")
    save_to_s3("emb/", args.s3_path)
