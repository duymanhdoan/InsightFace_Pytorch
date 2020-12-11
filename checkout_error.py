from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import os
from config import *



def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not os.path.exists(rootdir):
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):

        _bin = bins[i]
        print('->>>>>>>>>>>>>>>>> ',_bin)
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list



if __name__ == '__main__':
    test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf = get_config()
    root_paths = os.path.join(conf.data_path,args.rec_path)
    # print('=>>>>>>>>>>>>> {}'.format(os.path.exists(root_paths)))
    # print('type of path: {} -> path: {}'.format(type(root_paths),root_paths))
    # print('load mx rec from paths {}'.format(root_paths))
    # load_mx_rec(root_paths)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    print('load bin files from load bin \n')
    for bin_file in bin_files:
        full_path = os.path.join(root_paths, bin_file + '.bin')
        folder_path = full_path.split('.')[-2]
        print("full path: {} - status: {}".format(full_path, os.path.exists(full_path)))
        print("folder path: {} - status: {}".format(folder_path, os.path.exists(folder_path)))
        # load_bin(full_path,folder_path,test_transform)
