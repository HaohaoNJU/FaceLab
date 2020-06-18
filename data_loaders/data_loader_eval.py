# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> data_loader_eval.py
@Date : Created on 2020/4/29 14:23
@author: wanghao
========================"""
import sys
import os
import torch.utils.data as data
import numpy as np
import cv2
from scipy import misc
from numpy.random import randint, shuffle, choice
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle as pkl
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
import torch.backends.cudnn as cudnn
sys.path.append(os.path.abspath(".."))
def get_all_images(root_path,tails=["jpg","png","JPG","PNG","jpeg","JPEG"]):
    """
        Get all the images paths under the root_path
    """
    paths = []
    names = []
    for root_dir, dirs,files in os.walk(root_path):
        for file in files:
            if file[-3:] in tails:
                paths.append(os.path.join(root_dir,file))
                names.append(file)
    return paths,names

class Dataset(data.Dataset):
    def __init__(self,data_path=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        if isinstance(data_path,str):
            self.img_paths,_ = get_all_images(data_path)
        elif isinstance(data_path,list):
            self.img_paths = data_path
        else:
            raise TypeError("Can not recognize key value 'data_path'")
    def __getitem__(self,index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if len(img.split()) != 3:
            img = img.convert("RGB")
        img = self.transform(img)
        return img, img_path
    def __len__(self):
        return len(self.img_paths)

class Data_Loader():
    def __init__(self,data_path=None,batch_size=64,img_size=112,
                num_workers=4,pin_memory=True,shuffle=False,**kwargs):
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_classes = None
        self.data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
    ])
        self.build_dataset()
        self.build_loader()
    def reload(self):
        print("Rebuilding loader . . .")
        self.build_loader()
    def __len__(self):
        return self.sample_dataset.__len__()//self.batch_size
    def build_dataset(self):
        self.sample_dataset = Dataset(self.data_path,self.data_transform)
    def build_loader(self):
        self.sample_loader = torch.utils.data.DataLoader(dataset=self.sample_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=self.shuffle,
                                                    pin_memory=self.pin_memory,
                                                    num_workers=self.num_workers,
                                                    )

