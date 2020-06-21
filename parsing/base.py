###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample

import  parsing.resnet as resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet', 'MultiEvalModule', 'MultiEvalModuleCityscapes']

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 mean=[.485, .456, .406], std=[.229, .224, .225], root='~/.encoding/models'):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'mobilenet':
            self.pretrained = resnet.mobilenet(pretrained=False, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101coco':
            self.pretrained = resnet.resnet101coco(pretrained=False, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone == 'mobilenet':
            x = self.pretrained.features(x)
            c1 = x
            c2 = x
            c3 = x
            c4 = x
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


