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
import functools
import DAENet
from PIL import Image
import pdb

def takeSecond(elem):
    return elem[1]


def channel_1toN(img, num_channel):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    img = (transform1(img) * 255.0).long()
    T = torch.LongTensor(num_channel, img.size(1), img.size(2)).zero_()
    mask = torch.LongTensor(img.size(1), img.size(2)).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    return T.float()

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

    def forward_cari_generation(self, x):
        z = self.photo_encoder(x)
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
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2).squeeze() * (1-label.float())+ \
                torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2).squeeze() * label.float())
        return loss_contrastive

net = SiameseNetwork()
net.load_state_dict(torch.load('siamese.pth.tar'))
net.cuda()

net.eval()
cari_names = []
cari_f = open('/data/home/wenqingchu/dataset/parsing/cari_annos/annos.txt')
cari_enc= np.zeros((len(cari_f.readlines()), 128))
cari_f = open('/data/home/wenqingchu/dataset/parsing/cari_annos/annos.txt')
for i, cari_name in enumerate(cari_f.readlines()):
    cari_name = cari_name.strip()
    img0 = Image.open('/data/home/wenqingchu/dataset/parsing/cari_annos/label/' + cari_name)
    img0 = img0.crop((12,12,244,244))
    img0 = img0.resize((256,256), Image.NEAREST)
    img0 = channel_1toN(img0, 10)

    img0 = img0.cuda()
    z1,output1 = net.forward_cari(img0.unsqueeze(0))
    cari_enc[i] = z1[0].detach().cpu().numpy()
    cari_names.append(cari_name)

photo_f = open('/data/home/wenqingchu/dataset/parsing/celeba.list')
photo_names = []
f = open('retrieve_test_celeba.list', 'w')
photo_enc= np.zeros((len(photo_f.readlines()), 128))
photo_f = open('/data/home/wenqingchu/dataset/parsing/celeba.list')
for i, photo_name in enumerate(photo_f.readlines()):
    if i % 100 == 0:
        print(i)
    photo_name = photo_name.strip()
    img0 = Image.open('/data/home/wenqingchu/dataset/parsing/CelebA-LQ-mask/' + photo_name)
    img0 = channel_1toN(img0, 10)
    img0 = img0.cuda()
    z1,output1 = net.forward_photo(img0.unsqueeze(0))
    photo_enc[i] = z1[0].detach().cpu().numpy()
    photo_names.append(photo_name)
    dist = []
    for j in range(len(cari_names)):
        dd = np.sqrt(np.sum(np.square(photo_enc[i] - cari_enc[j])))
        dist.append((cari_names[j],dd))
    dist.sort(key=takeSecond)
    f.write(photo_name)
    for j in range(3):
        f.write(',' + dist[j][0])
    f.write('\n')
f.close()






