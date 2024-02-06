import argparse
import os
import socket
import sys
import time
import urllib
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from urllib.request import urlretrieve

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import numpy as np
from sklearn.decomposition import PCA

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

socket.setdefaulttimeout(20)
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent',
                      'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36')]
urllib.request.install_opener(opener)

def save_to_s3(src_path, dst_path):
    cmd = 's3cmd put -r ' + src_path + ' ' + dst_path
    os.system(cmd)

def is_valid_jpg(path):
    try:
        with open(path, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == b'\xff\xd9'
    except Exception:
        return False


def download_photo(url, path):
    for epoch in range(10): 
        try:
            urlretrieve(url, path)
            return True, None, None
        except Exception:  
            time.sleep(0.5)
    return False, url, path


def download_photos(meta_path):
    data_dir = os.path.dirname(meta_path)
    photo_dir = os.path.join(os.path.join(data_dir, 'photos'),'photos')
    os.makedirs(photo_dir, exist_ok=True)

    try:
        print(f'## Read {meta_path}')
        df = pd.read_json(os.path.join(data_dir, 'photos.json'), orient='records', lines=True)
    except:
        print('## Please first running "data_process.py" to generate "photos.json"!!!')
        return

    print(f'## Start to download pictures and save them into {photo_dir}')
    pool = ThreadPoolExecutor()
    tasks = []
    for name, url in zip(df['business_id'], df['imUrl']):
        path = os.path.join(photo_dir, name + '.jpg')
        if not os.path.exists(path) or not is_valid_jpg(path):
            task = pool.submit(download_photo, url, path)
            tasks.append(task)

    failed = []
    for i, task in enumerate(as_completed(tasks)):
        res, url, path = task.result()
        if not res:
            failed.append((url, path))
        print(f'## Tried {i}/{len(tasks)} photos!', end='\r', flush=True)
    pool.shutdown()

    for url, path in failed:
        print(f'## Failed to download {url} to {path}')
    print(f'## {len(tasks) - len(failed)} images were downloaded successfully to {photo_dir}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--photos_json', dest='photos_json', default='music/photos.json')
    parser.add_argument('--src_dir', default='')
    parser.add_argument('--photo_path', default='music/photos/')
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
        "--batch_size",
        default=1024,
        type=int,
    )
    args = parser.parse_args()

    cmd = 's3cmd get -r ' + args.src_dir
    os.system(cmd)
    download_photos(args.photos_json)
    # save_to_s3(args.photo_path, args.s3_path + args.photo_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
    ])
    img_data = ImageFolderWithPaths(args.photo_path,transform=trans)  # our custom dataset
    # img_data = datasets.ImageFolder(img_dir, transform=trans)
    batch_size = args.batch_size
    img_loader = DataLoader(img_data, batch_size=batch_size)

    model = models.vgg16(pretrained=True)
    model.classifier=model.classifier[0]
    model.eval()
    model = model.to(device)
    img_name = []
    img_emb = []
    for idx, (data, labels, path) in enumerate(img_loader):
        print("progress:",idx,len(img_loader))
        data = data.to(device)
        embeddings = model(data).detach().numpy()
        path=list(path)
        for i in range(len(path)):
            path[i] = os.path.basename(path[i]).split(".")[0]
        img_name.extend(path)
        img_emb.extend(embeddings)
    img_name = np.array(img_name)
    img_emb = np.array(img_emb)
    pca = PCA(n_components=128)
    pca_result = pca.fit_transform(img_emb)
    final_array = np.c_[img_name,pca_result]

    print(final_array.shape)
    os.makedirs("emb", exist_ok=True)
    np.savetxt("emb/img_emb.csv", final_array, delimiter=',', fmt="%s")
    save_to_s3("emb/", args.s3_path)
