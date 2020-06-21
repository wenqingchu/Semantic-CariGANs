import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from siamese_dataset import SiameseDataset
import functools
import DAENet
import pdb


train_batch_size = 32
train_number_epochs = 400

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

norm_layer = get_norm_layer('batch')

class DenseEncoder(nn.Module):
    def __init__(self, nc=10, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2,False], f_activation=nn.Sigmoid, f_args=[], norm_layer=nn.BatchNorm2d):
        super(DenseEncoder, self).__init__()
        self.ndim = ndim
        self.main = nn.Sequential(
                # input is (nc) x 256 x 256
                nn.Conv2d(nc, ndf, 7, stride=2, padding=3),
                nn.BatchNorm2d(ndf),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),



                # input is (ndf) x 64 x 64
                DAENet.DenseBlockEncoder(ndf, 4, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args,norm_layer=norm_layer),

                # input is (ndf*2) x 32 x 32
                DAENet.DenseBlockEncoder(ndf*2, 6, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args,norm_layer=norm_layer),

                # input is (ndf*4) x 16 x 16
                DAENet.DenseBlockEncoder(ndf*4, 12, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args,norm_layer=norm_layer),

                # input is (ndf*8) x 8 x 8
                DAENet.DenseBlockEncoder(ndf*8, 24, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockEncoder(ndf*8, ndf*8, 2, activation=activation, args=args,norm_layer=norm_layer),

                # input is (ndf*8) x 4 x 4
                DAENet.DenseBlockEncoder(ndf*8, 16, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args,norm_layer=norm_layer),
                f_activation(*f_args),
                )

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.ndim)
        return output



class DenseDecoder(nn.Module):
    def __init__(self, nz=128, nc=10, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False],f_activation=nn.Hardtanh, f_args=[0,1], norm_layer=nn.BatchNorm2d):
        super(DenseDecoder, self).__init__()
        self.nz = nz
        self.main   = nn.Sequential(
                # input is Z, going into convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

                # state size. (ngf*8) x 4 x 4
                DAENet.DenseBlockDecoder(ngf*8, 16, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf*8, ngf*8, norm_layer=norm_layer),

                # state size. (ngf*8) x 8 x 8
                DAENet.DenseBlockDecoder(ngf*8, 24, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf*8, ngf*4, norm_layer=norm_layer),

                # state size. (ngf*4) x 16 x 16
                DAENet.DenseBlockDecoder(ngf*4, 12, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf*4, ngf*2, norm_layer=norm_layer),

                # state size. (ngf*2) x 32 x 32
                DAENet.DenseBlockDecoder(ngf*2, 6, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf*2, ngf, norm_layer=norm_layer),

                # state size. (ngf) x 64 x 64
                DAENet.DenseBlockDecoder(ngf, 4, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf, ngf, norm_layer=norm_layer),


                # state size. (ngf) x 128 x 128
                DAENet.DenseBlockDecoder(ngf, 2, norm_layer=norm_layer),
                DAENet.DenseTransitionBlockDecoder(ngf, ngf, norm_layer=norm_layer),


                # state size. (ngf) x 256 x 256
                norm_layer(ngf),
                activation(*args),
                nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
                f_activation(*f_args),
                )

    def forward(self, inputs):
        output = self.main(inputs.view(-1, self.nz, 1, 1))
        return output


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.photo_encoder = DenseEncoder()
        self.cari_encoder = DenseEncoder()
        self.photo_decoder = DenseDecoder()
        self.cari_decoder = DenseDecoder()

    def forward_photo(self, x):
        z = self.photo_encoder(x)
        output = self.photo_decoder(z)
        return z, output

    def forward_cari(self, x):
        z = self.cari_encoder(x)
        output = self.cari_decoder(z)
        return z, output

    def forward(self, input1, input2):
        z1, output1 = self.forward_photo(input1)
        z2, output2 = self.forward_cari(input2)
        return z1, output1, z2, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2).squeeze() * (label.float())+ \
                torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2).squeeze() * (1 -label.float()))
        return loss_contrastive

siamese_dataset = SiameseDataset()
train_dataloader = DataLoader(siamese_dataset,
        shuffle=True,
        num_workers=8,
        batch_size=train_batch_size)


net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
criterion_recon = nn.L1Loss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)

counter = []
loss_history = []
iteration_number= 0

for epoch in range(0,train_number_epochs):
    test_dataset = SiameseDataset(mode='Val')
    test_dataloader = DataLoader(test_dataset,shuffle=True,num_workers=8,batch_size=1)
    net.eval()
    test_loss = 0
    for i, data in enumerate(test_dataloader):
        img0, img1, label = data['A'], data['B'], data['label']
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        z1, output1, z2, output2 = net(img0,img1)
        loss_recon = 0.01 * torch.sum(torch.abs(img0 - output1)) + 0.01 * torch.sum(torch.abs(img1 - output2))
        loss_contrastive = 100 * criterion(z1,z2,label)
        test_loss  = test_loss + 1.0 * loss_contrastive.item()  + 0.01 * loss_recon.item()
    print("Epoch: %d Test loss %f" % (epoch, test_loss))
    net.train()
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data['A'], data['B'], data['label']
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        z1, output1, z2, output2 = net(img0,img1)
        loss_contrastive = 100 * criterion(z1,z2,label)
        loss_recon = 0.01 * torch.sum(torch.abs(img0 - output1)) + 0.01 * torch.sum(torch.abs(img1 - output2))
        total_loss  = 1.0 * loss_contrastive  + 0.01 * loss_recon
        total_loss.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch: %d Iteration: %d Contrastive loss %f Reconstruction loss %f" % (epoch,i,loss_contrastive.item(), loss_recon.item()))
    torch.save(net.state_dict(), 'siamese.pth.tar')
