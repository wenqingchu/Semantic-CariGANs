#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import pdb
import torch.nn as nn
from psp import PSP
import pdb

palette = np.array([0, 0, 0, 244, 35, 232, 70, 70, 70, 102, 102, 156,190, 153, 153, 255, 0, 0, 250, 170, 30,0, 0, 230, 0, 80, 100, 152, 251, 152, 0, 255,255, 0, 0, 142, 119, 11, 32])

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette = np.append(palette, 0)


n_classes = 11
net = PSP(n_classes, 'resnet50')
net.load_state_dict(torch.load('models/19999_iter.pth'))
net.cuda()
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
resize = nn.Upsample(size=(256,256), mode='bilinear',align_corners=True)
with torch.no_grad():
    image_path = 'Scarlett_Johansson_C00001.jpg'
    img = Image.open(image_path)
    image_384 = img.resize((384, 384), Image.BILINEAR)
    img_384 = to_tensor(image_384)
    img_384 = torch.unsqueeze(img_384, 0)
    img_384 = img_384.cuda()
    out_384, _ = net(img_384)
    out_384 = resize(out_384)

    image_flip = image_384.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip = to_tensor(image_flip)
    img_flip = torch.unsqueeze(img_flip, 0)
    img_flip = img_flip.cuda()
    out_flip, _ = net(img_flip)
    out_flip_384 = resize(out_flip)

    image_256 = img.resize((256, 256), Image.BILINEAR)
    img_256 = to_tensor(image_256)
    img_256 = torch.unsqueeze(img_256, 0)
    img_256 = img_256.cuda()
    out_256, _ = net(img_256)
    out = (out_384 + out_256) / 4

    image_flip = image_256.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip = to_tensor(image_flip)
    img_flip = torch.unsqueeze(img_flip, 0)
    img_flip = img_flip.cuda()
    out_flip, _ = net(img_flip)
    out_flip_256 = resize(out_flip)
    out_flip = (out_flip_384 + out_flip_256) / 4

    parsing = out.squeeze(0).cpu().numpy()
    parsing_flip = out_flip.squeeze(0).cpu().numpy()
    parsing_flip = np.flip(parsing_flip, 2)
    parsing_flip_tmp = parsing_flip.copy()
    parsing_flip_tmp[2] = parsing_flip[3]
    parsing_flip_tmp[3] = parsing_flip[2]
    parsing_flip_tmp[4] = parsing_flip[5]
    parsing_flip_tmp[5] = parsing_flip[4]

    parsing = parsing + parsing_flip_tmp
    parsing = parsing.argmax(0)
    fg_pos = np.where(parsing==10)
    parsing[fg_pos[0],fg_pos[1]] = 0





    parsing = parsing.astype(np.uint8)
    parsing = Image.fromarray(parsing)
    parsing.putpalette(palette.tolist())
    parsing.save(image_path.replace('jpg', 'png'))

