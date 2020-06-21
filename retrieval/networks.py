import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import itertools
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable

import pdb
###############################################################################
# Helper Functions
###############################################################################
from torch.nn import Module, AdaptiveAvgPool2d



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
        inputs :
        x : input feature maps( B X C X W X H)
        returns :
        out : self attention value + input feature
        attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out,attention

class PyramidPooling(Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        #self.pool3 = AdaptiveAvgPool2d(4)
        #self.pool4 = AdaptiveAvgPool2d(8)

        out_channels = int(in_channels/2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w))
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w))
        return torch.cat((x, feat1, feat2), 1)



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


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if hasattr(net, 'which_model_netG'):
        which_model_netG = net.which_model_netG
        fc2_bias = net.fc2_bias
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, gain=init_gain)
    if hasattr(net, 'which_model_netG'):
        if which_model_netG == 'unbounded_stn' or which_model_netG == 'bounded_stn' or which_model_netG == 'affine_stn':
            net.fc2.bias.data.copy_(fc2_bias)
            net.fc2.weight.data.zero_()
    return net




class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU,args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                norm_layer(n_channels),
                activation(*args),
                nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
                )

    def forward(self, inputs):
        return self.main(inputs)


class waspDenseEncoder(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf = 32, ndim = 128,activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Sigmoid,f_args=[], norm_layer=nn.BatchNorm2d):
        super(waspDenseEncoder, self).__init__()
        self.ngpu = ngpu
        self.ndim = ndim

        self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                DenseBlockEncoder(ndf, 6, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2,activation=activation, args=args, norm_layer=norm_layer),

                DenseBlockEncoder(ndf*2, 12, norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2,activation=activation, args=args, norm_layer=norm_layer),

                DenseBlockEncoder(ndf*4, 24,norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2,activation=activation, args=args, norm_layer=norm_layer),

                DenseBlockEncoder(ndf*8, 16,norm_layer=norm_layer),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4,activation=activation, args=args, norm_layer=norm_layer),
                f_activation(*f_args),
                )

    def forward(self, input):
        output = self.main(input).view(-1,self.ndim)
        return output

class waspWarper(nn.Module):
    def __init__(self, imgSize = 256, batchSize = 1):
        super(waspWarper, self).__init__()
        self.batchSize = batchSize
        self.imgSize = imgSize

    def forward(self, input_img, input_grid):
        self.warp = input_grid.permute(0,2,3,1)
        self.output = F.grid_sample(input_img, self.warp)
        return self.output




class waspGridSpatialIntegral(nn.Module):
    def __init__(self, imgSize = 256, cuda = True):
        super(waspGridSpatialIntegral, self).__init__()
        self.w = imgSize
        self.filterx = torch.cuda.FloatTensor(1,1,1,self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1,1,self.w,1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if cuda:
            self.filterx.cuda()
            self.filtery.cuda()

    def forward(self, input_diffgrid):
        fullx = F.conv_transpose2d(input_diffgrid[:,0,:,:].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:,1,:,:].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:,:,0:self.w,0:self.w],fully[:,:,0:self.w,0:self.w]),1)
        return output_grid


class waspGridSpatialIntegral2(nn.Module):
    def __init__(self, imgSize=256, cuda=True):
        super(waspGridSpatialIntegral2, self).__init__()
        self.w = imgSize
        self.filterx = torch.cuda.FloatTensor(1, 1, 1, self.w).fill_(1)
        self.filtery = torch.cuda.FloatTensor(1, 1, self.w, 1).fill_(1)
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = Variable(self.filtery, requires_grad=False)
        if cuda:
            self.filterx.cuda()
            self.filtery.cuda()

    def forward(self, input_diffgrid):
        fullx = F.conv_transpose2d(input_diffgrid[:, 0, :, :].unsqueeze(1), self.filterx, stride=1, padding=0)
        fully = F.conv_transpose2d(input_diffgrid[:, 1, :, :].unsqueeze(1), self.filtery, stride=1, padding=0)
        output_grid = torch.cat((fullx[:, :, 0:self.w, -self.w:],fully[:, :, -self.w:, 0:self.w]), 1)
        return output_grid



class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                norm_layer(n_channels),
                activation(*args),
                nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False], norm_layer=nn.BatchNorm2d):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                norm_layer(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
                )

    def forward(self, inputs):
        return self.main(inputs)


class waspDenseDecoder(nn.Module):
    def __init__(self, ngpu=1, nz=128, nc=1, ngf=32, lb=0, ub=1,activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh,f_args=[0,1], norm_layer=nn.BatchNorm2d):
        super(waspDenseDecoder, self).__init__()
        self.ngpu   = ngpu
        self.main   = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

                DenseBlockDecoder(ngf*8, 16, norm_layer=norm_layer),
                DenseTransitionBlockDecoder(ngf*8, ngf*4, norm_layer=norm_layer),

                DenseBlockDecoder(ngf*4, 24, norm_layer=norm_layer),
                DenseTransitionBlockDecoder(ngf*4, ngf*2, norm_layer=norm_layer),

                DenseBlockDecoder(ngf*2, 12, norm_layer=norm_layer),
                DenseTransitionBlockDecoder(ngf*2, ngf, norm_layer=norm_layer),

                DenseBlockDecoder(ngf, 6, norm_layer=norm_layer),
                DenseTransitionBlockDecoder(ngf, ngf, norm_layer=norm_layer),

                norm_layer(ngf),
                activation(*args),
                nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
                f_activation(*f_args),
                )

    def forward(self, inputs):
        return self.main(inputs)


class Dense_DecodersIntegralWarper2(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, wdim = 128,imgSize=256, batch_size=1, norm_layer=nn.BatchNorm2d):
        super(Dense_DecodersIntegralWarper2, self).__init__()
        self.imagedimension = imgSize
        self.ngpu = ngpu
        self.wdim = wdim
        self.decoderW_left = waspDenseDecoder(ngpu=self.ngpu,nz=wdim, nc=2, ngf=ngf, lb=0, ub=1, activation=nn.Tanh, args=[],f_activation=nn.Sigmoid, f_args=[], norm_layer=norm_layer)
        self.decoderW_right = waspDenseDecoder(ngpu=self.ngpu,nz=wdim, nc=2, ngf=ngf, lb=0, ub=1, activation=nn.Tanh, args=[],f_activation=nn.Sigmoid, f_args=[], norm_layer=norm_layer)
        self.decoderW_right = waspDenseDecoder(ngpu=self.ngpu,nz=wdim, nc=2, ngf=ngf, lb=0, ub=1, activation=nn.Tanh, args=[],f_activation=nn.Sigmoid, f_args=[], norm_layer=norm_layer)

        self.warper   = waspWarper(imgSize, batch_size)
        self.integrator = waspGridSpatialIntegral(imgSize=imgSize)
        self.integrator2 =waspGridSpatialIntegral2(imgSize=imgSize)
        self.cutter = nn.Hardtanh(-1,1)

    def forward(self, zW):
        self.diffentialWarping_left = (self.decoderW_left(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) +2.0 / self.imagedimension
        self.diffentialWarping_right = (self.decoderW_right(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) + 2.0 / self.imagedimension

        self.warping_left = self.integrator(self.diffentialWarping_left)-1.0
        self.warping_right = 1.0 - self.integrator2(self.diffentialWarping_right)
        self.warping_left = self.cutter(self.warping_left)
        self.warping_right = self.cutter(self.warping_right)
        self.warping = (self.warping_left + self.warping_right) /2.0 / 63.0 * 64.0
        return self.warping


