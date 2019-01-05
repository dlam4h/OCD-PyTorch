# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class OCD(nn.Module):
    def __init__(self, input_channel = 3, cls_num = 1):
        super(OCD, self).__init__()
        self.encoder = torchvision.models.vgg16(pretrained=True).features

        self.conv1 = self.encoder[0]
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = self.encoder[2]
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv3 = self.encoder[5]
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = self.encoder[7]
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxPool2 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv5 = self.encoder[10]
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = self.encoder[12]
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_1 = self.encoder[14]
        self.conv6_1_bn = nn.BatchNorm2d(256)
        self.maxPool3 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv7 = self.encoder[17]
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = self.encoder[19]
        self.conv8_bn = nn.BatchNorm2d(512)
        self.maxPool4 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        self.conv9_bn = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv10.weight)
        nn.init.constant_(self.conv10.bias, 0)
        self.conv10_bn = nn.BatchNorm2d(1024)
 

        self.upsampconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
 
        self.conv11 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv11.weight)
        nn.init.constant_(self.conv11.bias, 0)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv12.weight)
        nn.init.constant_(self.conv12.bias, 0)
        self.conv12_bn = nn.BatchNorm2d(512)
 
        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
 
        self.conv13 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv13.weight)
        nn.init.constant_(self.conv13.bias, 0)
        self.conv13_bn = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv14.weight)
        nn.init.constant_(self.conv14.bias, 0)
        self.conv14_bn = nn.BatchNorm2d(256)
 
        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
 
        self.conv15 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv15.weight)
        nn.init.constant_(self.conv15.bias, 0)
        self.conv15_bn = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv16.weight)
        nn.init.constant_(self.conv16.bias, 0)
        self.conv16_bn = nn.BatchNorm2d(128)
 
        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
 
        self.conv17 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv17.weight)
        nn.init.constant_(self.conv17.bias, 0)
        self.conv17_bn = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv18.weight)
        nn.init.constant_(self.conv18.bias, 0)
        self.conv18_bn = nn.BatchNorm2d(64)
 
        self.conv19 = nn.Conv2d(64, cls_num, 1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv19.weight)
        nn.init.constant_(self.conv19.bias, 0)
        # self.conv19_bn = nn.BatchNorm2d(cls_num)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x_copy1_2 = x
        x = self.maxPool1(x)
 
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x_copy3_4 = x
        x = self.maxPool2(x)
 
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv6_1_bn(self.conv6_1(x)))
        x_copy5_6 = x
        x = self.maxPool3(x)
 
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.dropout(x, 0.5)
        x_copy7_8 = x
        x = self.maxPool4(x)
 
        x = F.relu(self.conv9_bn(self.conv9(x)))
        x = F.relu(self.conv10_bn(self.conv10(x)))
        x = F.dropout(x, 0.5)

        x = F.relu(self.upsampconv1(x))
        x = torch.cat((x, x_copy7_8), 1)
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
 
        x = F.relu(self.upsampconv2(x))
        x = torch.cat((x, x_copy5_6), 1)
        x = F.relu(self.conv13_bn(self.conv13(x)))
        x = F.relu(self.conv14_bn(self.conv14(x)))
 
        x = F.relu(self.upsampconv3(x))
        x = torch.cat((x, x_copy3_4), 1)
        x = F.relu(self.conv15_bn(self.conv15(x)))
        x = F.relu(self.conv16_bn(self.conv16(x)))
 
        x = F.relu(self.upsampconv4(x))
        x = torch.cat((x, x_copy1_2), 1)
        x = F.relu(self.conv17_bn(self.conv17(x)))
        x = F.relu(self.conv18_bn(self.conv18(x)))

        x = self.conv19(x)
        return x
