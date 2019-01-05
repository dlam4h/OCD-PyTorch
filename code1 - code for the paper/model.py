# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class OCD(nn.Module):
    def __init__(self, input_channel = 3, cls_num = 1):
        super(OCD, self).__init__()
        self.features = nn.Sequential(  nn.Conv2d(3, 64, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(64, 128, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 128, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(128, 256, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(256, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2, 2))
        
        self.conv14 = nn.Conv2d(512, 4096, 7, stride=1, padding=3)
        nn.init.xavier_uniform_(self.conv14.weight)
        nn.init.constant_(self.conv14.bias, 0.1)
        # self.conv14_bn = nn.BatchNorm2d(4096)

        self.conv15 = nn.Conv2d(4096, 512, 1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv15.weight)
        nn.init.constant_(self.conv15.bias, 0.1)
        # self.conv15_bn = nn.BatchNorm2d(512)

        self.upsampconv1 = nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0)
        
        self.conv16 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv16.weight)
        nn.init.constant_(self.conv16.bias, 0.1)
        # self.conv16_bn = nn.BatchNorm2d(512)

        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.conv17 = nn.Conv2d(256, 256, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv17.weight)
        nn.init.constant_(self.conv17.bias, 0.1)
        # self.conv17_bn = nn.BatchNorm2d(256)

        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.conv18 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv18.weight)
        nn.init.constant_(self.conv18.bias, 0.1)
        # self.conv18_bn = nn.BatchNorm2d(128)

        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv19 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv19.weight)
        nn.init.constant_(self.conv19.bias, 0.1)
        # self.conv19_bn = nn.BatchNorm2d(64)

        self.upsampconv5 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0)
        self.conv20 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv20.weight)
        nn.init.constant_(self.conv20.bias, 0.1)
        # self.conv20_bn = nn.BatchNorm2d(32)
 
        self.conv21 = nn.Conv2d(32, cls_num, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv21.weight)
        nn.init.constant_(self.conv21.bias, 0.1)

    def forward(self, x):
        x = self.features(x)

        x = F.relu(self.conv14(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.conv15(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv1(x))

        x = F.relu(self.conv16(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv2(x))

        x = F.relu(self.conv17(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv3(x))

        x = F.relu(self.conv18(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv4(x))

        x = F.relu(self.conv19(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv5(x))

        x = F.relu(self.conv20(x))
        x = F.dropout(x, 0.5)
        x = self.conv21(x)
        return x