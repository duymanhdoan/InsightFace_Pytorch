import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import args
import numpy as np 
import pathlib
from torchvision import transforms as trans
from model import l2_norm
import args
import os 
from multiprocessing import Pool
import tqdm 


def extract_folder_image(path, conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = []
    embs = []
    for file in path.iterdir():
        if not file.is_file():
            continue
        else:
            try:
                img = Image.open(file)
                if img.size != (112, 112):
                    img = mtcnn.align(img)
                with torch.no_grad():
                    if tta:
                        mirror = trans.functional.hflip(img)
                        emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                        emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
            except: 
                continue
    if embs == []: 
        return None, None        
    embedding = torch.cat(embs).mean(0,keepdim=True)
    embeddings.append(embedding)
    names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    return embeddings, names

def extract_single_image(file,conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = []
    embs = []
    img = Image.open(file)
    try:
        if img.size != (112, 112):
            img = mtcnn.align(img)
        with torch.no_grad():
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    except: 
        print('load error')
    
    if len(embs) == 0:
        return None, None
    embedding = torch.cat(embs).mean(0,keepdim=True)
    embeddings.append(embedding)
    names.append(file.split('/')[-1])
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    return embeddings, names

def distance(source_embs, target_embs): 
    diff = source_embs.transpose(1,0).unsqueeze(0) - target_embs.transpose(1,0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > args.threshold] = -1 # if no match, set idx to -1
    return min_idx, minimum


def caculate_distace_foler(rootdir, parse_text_file):
    
    conf = get_config(False)
    mtcnn = MTCNN()
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, True, True)
    else:
        learner.load_state(conf, True, True)
    learner.model.eval()
    
    
    
    embedding = [] 
    names = []
    folder = os.listdir(rootdir) 
    for path in tqdm.tqdm(folder): 
        dirs = os.path.join(rootdir,path)
        try:
            embs, name = extract_folder_image(Path(dirs) ,conf , learner.model, mtcnn, tta= args.tta)
            embedding.append(embs)
            names.append(name)
        except: 
            continue
    # if os.path.exists(parse_text_file): 
    #     pass 
    # else: 
    with open(parse_text_file,"w") as fs: 
        for i in tqdm.tqdm(range(0,len(embedding)-1)): 
            for j in range(i+1,len(embedding)): 
                _, score = distance(embedding[i],embedding[j])
                print(score)
                line = 'distance class:{} -> class:{} is:{:.2f}\n'.format(names[i],names[j],score[0])
                fs.write(line)
            fs.write('       ===================== \n')

def caculate_distance_image(root_folder,parse_text_file): 
    conf = get_config(False)
    mtcnn = MTCNN()
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, True, True)
    else:
        learner.load_state(conf, True, True)
    learner.model.eval()
    
    embedding = [] 
    names = []
    
    folder = os.listdir(root_folder) 
    for file in folder: 
        try:
            img = os.path.join(root_folder,file)     
            embs, name = extract_single_image(img, conf, learner.model, mtcnn, tta= args.tta)
            embedding.append(embs)
            names.append(name)
        except: 
            continue
    if os.path.exists(parse_text_file): 
        pass 
    else: 
        with open(parse_text_file,"w") as fs: 
            fs.write('\n CLASS: {} \n'.format(root_folder.split('/')[-1]))
            for i in range(1,len(embedding),1):  
                embs1 = embedding[0]
                embs2 = embedding[i]
                _, score = distance(embs1, embs2)
                line = 'distance image:{} -> image:{} is:{:.4f} \n'.format(names[0],names[i],score[0])
                fs.write(line)
            
            for i in range(2,len(embedding)):      
                _, score = distance(embedding[1],embedding[i])
                line = 'distance image:{} -> image:{} is:{:.4f} \n'.format(names[1],names[i],score[0])
                fs.write(line)
                

if __name__ == '__main__':

    # path_folder = '/home/minglee/Documents/aiProjects/VN-celeb/1'
    root_text_file = '/mnt/DATA/duydmFabbi/model_train/evaluate_vn_celeb'
    rootdir = '/mnt/DATA/duydmFabbi/dataFace/VN-CELEB-DATASET/VN-celeb'
    
    for dirs in tqdm.tqdm(os.listdir(rootdir)):     
        path_sub_class = os.path.join(rootdir, dirs)
        parse_text_class = 'Class:{}.txt'.format(dirs)
        caculate_distance_image(path_sub_class,os.path.join(root_text_file, parse_text_class))

    folder_file =  'distance_folder.txt'
    root_folder_file = os.path.join(root_text_file, folder_file)
    caculate_distace_foler(rootdir, root_folder_file)













    
    
    