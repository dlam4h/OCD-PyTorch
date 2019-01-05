# -*- coding: UTF-8 -*-
import torch
from model import OCD
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from torchvision import models
from torchvision import transforms as T
import glob
import cv2

use_gpu = torch.cuda.is_available()
print(use_gpu)
if use_gpu:
    torch.cuda.set_device(0)
model = OCD()
model_weight = '2018-12-28 15-50-23 OCD_VOC2012100.pt'
if use_gpu:
    model.cuda()
    model.load_state_dict(torch.load(model_weight))
else:
    model.load_state_dict(torch.load(model_weight,map_location='cpu'))
# model.eval()
img = cv2.imread('./2007_000256.jpg',1)/255.0
# img[0:336,0:496,:]
# print(img.shape)
# new_img = np.zeros((512,352,3))
# new_img[0:500,0:334]=img
# img = img.resize((480,480))
img = np.transpose(img[0:336,0:496,:],(2,0,1))
img = torch.from_numpy(img).unsqueeze(0).float()
if use_gpu:
    img = img.cuda()
    outputs = model(img)
    outputs = F.sigmoid(outputs).squeeze().cpu().detach().numpy()
    # outputs = F.sigmoid(outputs).squeeze(0).squeeze().cpu().detach().numpy()
else:
    outputs = model(img)
    outputs = F.sigmoid(outputs).squeeze(0).squeeze().data.numpy()
# print(np.histogram(outputs))
result = outputs
result[result<0.5]=0
result[result>=0.5]=255
# dlam = Image.fromarray(result.astype(np.uint8))
cv2.imwrite('result.tif',result)