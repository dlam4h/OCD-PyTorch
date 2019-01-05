# -*- coding: UTF-8 -*-
import torch
import datetime
from torchvision import transforms as T
import os
from logger import LogWay
from config import opt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import OCD
from dataset_OCD import My_Dataset
from imageio import imread
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(0)
    print('Use GPU')
else:
    print('Use CPU')

dlamly=[1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001]

def adjusting_rate(optimizer,learning_rate,epoch):
    dlam = 10
    if epoch%dlam == 0:
        lr = learning_rate*dlamly[epoch//dlam]
        for parm_group in optimizer.param_groups:
            parm_group['lr']=lr

def train():
    model_list = [OCD]
    model_list_name = ['OCD']
    dataset_name = ['VOC2012']
    img_train = ['./image']
    mask_train = ['./label']

    for cs in range(len(model_list_name)):
        for data_index in range(len(dataset_name)):
            model = OCD(input_channel = opt.input_channel, cls_num = opt.cls_num)
            # model_dict = model.state_dict()
            # vgg16 = models.vgg16(pretrained=True)
            # pretrained_dict = vgg16.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            # for param in list(model.parameters())[:25]:
            #     print(param.shape)
            #     param.requires_grad = False

            model_name = model_list_name[cs]
            data_name = dataset_name[data_index]
            train_logger = LogWay(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +model_name+'_'+data_name+ '.txt')
            train_data = My_Dataset(img_train[data_index], mask_train[data_index])
            train_dataloader = DataLoader(train_data,batch_size = 1, shuffle = True, num_workers = 0)

            if opt.cls_num == 1:
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = torch.nn.NLLLoss()
            if use_gpu:
                model.cuda()
                if opt.cls_num == 1:
                    criterion = torch.nn.BCEWithLogitsLoss().cuda()
                else:
                    criterion = torch.nn.NLLLoss().cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, betas=(0.9, 0.99),weight_decay=0.0005)
            for epoch in range(opt.epoch):
                loss_sum=0
                for i,(data,target) in enumerate(train_dataloader):
                    data = Variable(data).float().squeeze()
                    target = Variable(target).float().squeeze()
                    if use_gpu:
                        data = data.cuda()
                        target = target.cuda()
                    outputs = model(data)
                    if opt.cls_num == 1:
                        outputs = outputs.view(-1)
                        mask_true = target.view(-1)
                        loss = criterion(outputs,mask_true)
                    else:
                        outputs = F.LogSoftmax(outputs, dim=1)
                        loss = criterion(outputs, target)
                    loss_sum = loss_sum + loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("dataset:{} epoch:{} batch:{} loss:{}".format(data_name,epoch+1,i,loss.item()))
                realepoch = epoch + 1
                if(realepoch%1==0):
                    info = 'Time:{} dataset:{} Epoch:{} Loss_avg:{}\n'.format(str(datetime.datetime.now()),data_name, epoch+1, loss_sum/(i+1))
                    train_logger.add(info)
                adjusting_rate(optimizer,opt.learning_rate,epoch+1)
                if(realepoch%20==0):
                    save_name = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')+' '+model_name+'_'+data_name+str(realepoch)+'.pt'
                    torch.save(model.state_dict(),save_name)
if __name__ == '__main__':
    train()