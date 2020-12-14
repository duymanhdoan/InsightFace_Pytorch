from pathlib import Path
from config import get_config
from data.data_pipe import load_bin, load_mx_rec
import argparse
import args
import os
if __name__ == '__main__':

    conf = get_config()
    root_paths = conf.data_path/args.rec_path
    print('load mx rec from paths {}'.format(root_paths))
    load_mx_rec(root_paths)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    print('load bin files from load bin \n')
    for bin_file in bin_files:
        full_path = root_paths / (bin_file + '.bin')
        folder_path = root_paths/ bin_file
        print("full path: {} - status: {}".format(full_path.exists()))
        print("folder path: {} - status: {}".format(folder_path.exists()))
        load_bin(full_path,folder_path,conf.test_transform)
