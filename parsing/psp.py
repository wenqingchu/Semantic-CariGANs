###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torch.nn.functional as F


from parsing.base import BaseNet
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d



class PyramidPooling(Module):
    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),norm_layer(out_channels),ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),norm_layer(out_channels),ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),norm_layer(out_channels),ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),norm_layer(out_channels),ReLU(True))


    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)




class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels,inter_channels, 3, padding=1, bias=False),norm_layer(inter_channels),nn.ReLU(), nn.Dropout2d(0.1, False),nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)



class PSP(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer)
        #self.head = PSPHead(1280, nclass, norm_layer)
        self.head = PSPHead(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = upsample(x, (h,w), mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return x, auxout



class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
        'cityscapes': 'cityscapes',
        'gta5': 'gta5',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation, ContextSegmentation, GTA5Segmentation
    model = PSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

