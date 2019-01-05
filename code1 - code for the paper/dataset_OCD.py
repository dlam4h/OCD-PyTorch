#-*- coding: UTF-8 -*-
import os
import torch
from imageio import imread, imwrite
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from skimage import img_as_ubyte
from config import opt
from PIL import Image, ImageEnhance
import random
import cv2
import warnings

warnings.filterwarnings('ignore')

class My_Dataset(data.Dataset):
    def __init__(self, img_path, mask_path):
        super(My_Dataset, self).__init__()
        image_list = [img for img in os.listdir(img_path)]
        self.big_imgs = [os.path.join(img_path, img) for img in image_list]
        self.big_masks = [os.path.join(mask_path, '{}{}'.format(img.split('.')[0], '.png')) for img in image_list]
        self.r_size = 224
        self.c_size = 224
        print("length:{}".format(len(self.big_imgs)))

    def __len__(self):
        return len(self.big_imgs)

    def __getitem__(self, item):
        Img = cv2.imread(self.big_imgs[item],1)/255.0
        Mask = cv2.imread(self.big_masks[item],0)/255.0
        # Mask[Mask != 255]=0
        # Mask[Mask == 255]=1 
        Img = np.transpose(Img,(2,0,1))
        # mean_pix = [103.939, 116.779, 123.68]
        # Img[0] = Img[0]-103.939
        # Img[1] = Img[1]-116.779
        # Img[2] = Img[2]-123.68
        r,c = Mask.shape
        imgs = np.zeros((8,3,self.r_size,self.c_size))
        masks = np.zeros((8,self.r_size,self.c_size))
        indices_r = random.sample(range(0,r-self.r_size), 2)
        indices_c = random.sample(range(0,c-self.c_size), 2)
        index = 0
        for i in indices_r:
            for j in indices_c:
                imgs[index,:,:,:] = Img[:,i:i+self.r_size,j:j+self.c_size]
                imgs[index+4,:,:,:] = Img[:,r:r-self.r_size-1:-1,c:c-self.c_size-1:-1]
                masks[index,:,:] = Mask[i:i+self.r_size,j:j+self.c_size]
                masks[index+4,:,:] = Mask[r:r-self.r_size-1:-1,c:c-self.c_size-1:-1]
                index = index +1
        # print(np.histogram(masks))
        data = torch.from_numpy(imgs)
        label = torch.from_numpy(masks)
        return data, label