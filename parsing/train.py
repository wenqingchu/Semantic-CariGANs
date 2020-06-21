#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
#from model import BiSeNet
from psp import PSP
from celeba_dataset import FaceMask
from loss import OhemCELoss
from optimizer import Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from PIL import Image

import os
import os.path as osp
import logging
import time
import datetime
import argparse
import pdb

respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def adjust_learning_rate(optimizer, epoch, step, max_step, args):
    lr = args.lr
    if step < 500 and epoch==0:
        lr = lr * float(1 + step ) / 500
    elif epoch > 5:
        lr = lr * (1. - float(step) / max_step) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = 0,
            )
    return parse.parse_args()



def train():
    args = parse_args()
    cudnn.benchmark = True
    # dataset
    #n_classes = 19
    n_classes = 11
    n_img_per_gpu = 32
    n_workers = 8
    #cropsize = [448, 448]
    cropsize = [224, 224]
    data_root = 'dataset/parsing/'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    dl = DataLoader(ds, batch_size=n_img_per_gpu, num_workers=8, shuffle=True, drop_last=True)
    # model
    ignore_idx = -100
    #net = BiSeNet(n_classes=n_classes)
    net = PSP(11, 'resnet50')
    #net.load_state_dict(torch.load('model_best.pth.tar')["state_dict"])

    net = net.cuda()


    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//4
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).cuda()
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx).cuda()

    ## optimizer
    #momentum = 0.9
    weight_decay = 1e-4
    lr_start = 2e-3
    args.lr = lr_start
    max_iter = 120000
    train_optimizer = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=weight_decay)

    ## train loop
    msg_iter = 10
    loss_avg = []
    diter = iter(dl)
    epoch = 0

    net.train()
    for it in range(max_iter):
        adjust_learning_rate(train_optimizer, epoch, it, max_iter, args)
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        train_optimizer.zero_grad()
        out, out16 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss = lossp + loss2
        loss.backward()
        train_optimizer.step()
        loss_avg.append(loss.item())



        #  print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            #lr = train_optimizer.lr
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'loss: {loss:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    loss = loss_avg,
                )
            print(msg)
            loss_avg = []
        if (it+1) % 10000 == 0:
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state, './res/cp/{}_iter.pth'.format(it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()
