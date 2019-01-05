# -*- coding: UTF-8 -*-
class DefaultConfig(object):
    env = 'default'
    epoch = 100 #迭代次数
    lr_decay = 0.95   
    weight_decay = 2e-4
    momentum = 0.9
    input_channel = 3 #输入图片的通道数
    cls_num = 1 #输出图片的通道数
    
    learning_rate = 0.01 #初始学习率
opt =DefaultConfig()
