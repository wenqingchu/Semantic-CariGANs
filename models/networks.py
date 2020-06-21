import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import itertools
from .tps_grid_gen import TPSGridGen
from .grid_sample import grid_sample
import torch.nn.functional as F
from .inverse_tps_grid_gen import InverseTPSGridGen

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


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], num_style = 8):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resnet_6blocks_unit':
        net = ResnetGeneratorUnit(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, num_style=num_style)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def getBaseGrid(N=64, normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N-1), (N), 2.0)
    if normalize:
        a = a/(N-1.0)
    x = a.repeat(N,1)
    y = x.t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid.float()



def define_S(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'tps_gan':
        net = TpsGenerator(input_nc, output_nc, 'bounded_stn', 64, 64, ngf=32, norm_layer=norm_layer)
    elif netG == 'inverse_tps_gan':
        net = InverseTpsGenerator(input_nc, output_nc, 'bounded_stn', 64, 64, ngf=32, norm_layer=norm_layer)
    elif netG == 'star_gan':
        net = StarGenerator(ngpu=1, nc=input_nc, ngf=32, ndf=32, idim=16, wdim=119, zdim=128, imgSize=64, batch_size=1, norm_layer=norm_layer)
    elif netG == 'parseref_gan':
        net = ParserefGenerator(ngpu=1, nc=input_nc, ngf=32, ndf=32, zdim=128, rdim=128, imgSize=64, batch_size=1, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'dilated':
        net = DilatedDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


class LocationAwareReconstructedLoss(nn.Module):
    def __init__(self, imgSize = 256, cuda = True):
        super(LocationAwareReconstructedLoss, self).__init__()
        self.w = imgSize
        self.criterion = nn.MSELoss()
        self.filterx = torch.cuda.FloatTensor(1, 11, 1, self.w).fill_(0)
        single_filterx = torch.arange(self.w) / 256.0
        for i in range(10):
            self.filterx[0][i+1][0] = single_filterx
        self.filterx = Variable(self.filterx, requires_grad=False)
        self.filtery = torch.cuda.FloatTensor(1, 11, self.w, 1).fill_(0)
        single_filtery = torch.arange(self.w) / 256.0
        for i in range(10):
            self.filtery[0][i + 1] = single_filtery.view(self.w, -1)
        self.filtery = Variable(self.filtery, requires_grad=False)
        self.filterx.cuda()
        self.filtery.cuda()

    def forward(self, x, y, weight=1):
        x_x = F.avg_pool2d(x, kernel_size=(1, self.w))
        x_y = F.avg_pool2d(x, kernel_size=(self.w, 1))
        fullx = F.conv2d(x_y, self.filterx) + F.conv2d(x_x, self.filtery)
        y_x = F.avg_pool2d(y, kernel_size=(1, self.w))
        y_y = F.avg_pool2d(y, kernel_size=(self.w, 1))
        fully = F.conv2d(y_y, self.filterx) + F.conv2d(y_x, self.filtery)
        self.loss = torch.sum(torch.mul(fullx - fully, fullx - fully))
        return self.loss


class BiasReduceLoss(nn.Module):
    def __init__(self,opt):
        super(BiasReduceLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.L1Loss()

    def forward(self, x, y, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.avg = torch.mean(x,0).unsqueeze(0)
        self.loss = w*self.criterion(self.avg, y)
        return self.loss

class TotalVaryLoss(nn.Module):
    def __init__(self,opt):
        super(TotalVaryLoss, self).__init__()
        self.opt = opt

    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w * (torch.sum(torch.abs(x[:, :, :, :-1] -x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        return self.loss


class SelfSmoothLoss2(nn.Module):
    def __init__(self,opt):
        super(SelfSmoothLoss2, self).__init__()
        self.opt = opt

    def forward(self, x, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        self.y_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
        self.loss = torch.sum(torch.mul(self.x_diff, self.x_diff)) + torch.sum(torch.mul(self.y_diff, self.y_diff))
        self.loss = w * self.loss
        return self.loss





# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class TpsGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, which_model_netG, image_width, image_height, ngf=32,
                     norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, padding_type='reflect', gan='vanilla'):
        assert (n_blocks >= 0)
        super(TpsGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_dropout = use_dropout
        self.which_model_netG = which_model_netG
        span_range = 0.9
        span_range_height = span_range
        span_range_width = span_range
        r1 = span_range_height
        r2 = span_range_width
        grid_height = 8
        grid_width = 8
        self.image_height = image_height
        self.image_width = image_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
            )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == None:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    nn.ReLU(True)]
        else:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                        bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
        # n_downsampling = 2
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            if norm_layer == None:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                    stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)]
        model += [nn.AdaptiveAvgPool2d(grid_height)]

        self.model = nn.Sequential(*model)
        self.fc1 = nn.Linear(grid_width * grid_height * ngf * mult, 200)
        if which_model_netG == 'affine_stn':
            self.fc2 = nn.Linear(200, 6)
        else:
            self.fc2 = nn.Linear(200, 2 * grid_width * grid_width)
        if which_model_netG == 'unbounded_stn':
            bias = target_control_points.view(-1)
        elif which_model_netG == 'bounded_stn':
            bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
            bias = bias.view(-1)
        elif which_model_netG == 'affine_stn':
            bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)

        self.fc2_bias = bias
        self.tps = TPSGridGen(image_height, image_width, target_control_points)

    def forward(self, input):
        batch_size = input.size(0)
        x = self.model(input)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.which_model_netG == 'affine_stn':
            theta = x.view(-1, 2, 3)
            grid = F.affine_grid(theta, input.size())
            transformed_x = F.grid_sample(input, grid, padding_mode='border')
            return transformed_x, theta
        if self.which_model_netG == 'bounded_stn':
            points = F.tanh(x)
        elif self.which_model_netG == 'unbounded_stn':
            points = x
        #print(points[0].detach().cpu())
        source_control_points = points.view(batch_size, -1, 2)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.image_height, self.image_width, 2)
        return grid, source_control_points

def print_grad(grad):
    print('target_control_point_offset_grad')
    print(grad.data.cpu())

class InverseTpsGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, which_model_netG, image_width, image_height, ngf=32,
                     norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, padding_type='reflect', gan='vanilla'):
        assert (n_blocks >= 0)
        super(InverseTpsGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_dropout = use_dropout
        span_range = 0.9
        span_range_height = span_range
        span_range_width = span_range
        r1 = span_range_height
        r2 = span_range_width
        grid_height = 8
        grid_width = 8
        self.image_height = image_height
        self.image_width = image_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
            )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == None:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    nn.ReLU(True)]
        else:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                        bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            if norm_layer == None:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                    stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)]
        model += [nn.AdaptiveAvgPool2d(grid_width)]

        self.model = nn.Sequential(*model)
        self.fc1 = nn.Linear(grid_width * grid_width * ngf * mult, 200)
        if which_model_netG == 'affine_stn':
            self.fc2 = nn.Linear(200, 6)
        else:
            self.fc2 = nn.Linear(200, 2 * 17)
        if which_model_netG == 'unbounded_stn':
            bias = target_control_points.view(-1)
        elif which_model_netG == 'bounded_stn':
            bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
            bias = bias.view(-1)
        elif which_model_netG == 'affine_stn':
            bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)

        self.fc2_bias = bias

    def forward(self, input, source_control_points):
        batch_size = input.size(0)
        x = self.model(input)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        points = F.hardtanh(x, -1, 1)
        target_control_points_offset = points.view(batch_size, -1, 2)
        tps = InverseTPSGridGen()
        source_control_points = source_control_points.view(batch_size, -1, 2)
        target_control_points = source_control_points + 0.25 * target_control_points_offset
        source_coordinate_list = []
        for i in range(batch_size):
            source_coordinate = tps(self.image_height, self.image_width, source_control_points[i].unsqueeze(0), target_control_points[i])
            source_coordinate_list.append(source_coordinate)
        source_coordinate = torch.cat(tuple(source_coordinate_list), 0)
        grid = source_coordinate.view(batch_size, self.image_height, self.image_width, 2)

        return grid, target_control_points


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
        self.diffentialWarping_left = (self.decoderW_left(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) + 2.0 / self.imagedimension
        self.diffentialWarping_right = (self.decoderW_right(zW.view(-1, self.wdim, 1, 1)) - 0.5) * (4.0 / self.imagedimension) + 2.0 / self.imagedimension

        self.warping_left = self.integrator(self.diffentialWarping_left)-1.0
        self.warping_right = 1.0 - self.integrator2(self.diffentialWarping_right)
        self.warping_left = self.cutter(self.warping_left)
        self.warping_right = self.cutter(self.warping_right)
        self.warping = (self.warping_left + self.warping_right) /2.0 / 63.0 * 64.0
        return self.warping






class StarGenerator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, idim = 16, wdim= 128, zdim = 128, imgSize = 256, batch_size=1, norm_layer=nn.BatchNorm2d):
        super(StarGenerator, self).__init__()
        self.encoders = waspDenseEncoder(ngpu=ngpu, nc=nc, ndf=ndf,ndim=zdim, norm_layer=norm_layer)
        self.decoders = Dense_DecodersIntegralWarper2(ngpu, nc, ngf,ndf, wdim+9, imgSize, batch_size, norm_layer=norm_layer)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        input_c = c.view(c.size(0), c.size(1), 1, 1)
        input_c = input_c.repeat(1, 1, x.size(2), x.size(3))
        dp0_img = torch.cat([x, input_c], dim=1)

        dp0_zW = torch.cat([dp0_z, c], dim=1)
        dp0_Wact = self.decoders(dp0_zW)
        dp0_Wact = dp0_Wact.permute(0, 2, 3, 1)
        return dp0_Wact


class ParserefGenerator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ngf=32, ndf=32, zdim = 128, rdim= 128, imgSize = 256, batch_size=1, norm_layer=nn.BatchNorm2d):
        super(ParserefGenerator, self).__init__()
        self.cari_encoders = waspDenseEncoder(ngpu=ngpu, nc=nc, ndf = ndf, ndim = zdim, norm_layer=norm_layer)
        self.photo_encoders = waspDenseEncoder(ngpu=ngpu, nc=nc, ndf = ndf, ndim = rdim, norm_layer=norm_layer)
        self.cari_decoders = Dense_DecodersIntegralWarper2(ngpu, nc, ngf, ndf, zdim + rdim, imgSize, batch_size, norm_layer=norm_layer)
        self.photo_decoders = Dense_DecodersIntegralWarper2(ngpu, nc, ngf, ndf, zdim + rdim, imgSize, batch_size, norm_layer=norm_layer)

    def forward(self, photo, cari1, is_cari=True, is_blend=False,cari2=None, cari3=None):
        # Replicate spatially and concatenate domain information.
        dp0_photo = self.photo_encoders(photo)
        dp0_cari1 = self.cari_encoders(cari1)
        dp0_zr = torch.cat([dp0_cari1, dp0_photo], dim=1)
        #for test only
        #dp0_zr = torch.cat([(dp0_photo*0.2 + dp0_cari1*0.8),dp0_photo], dim=1)
        if is_blend:
            dp0_cari0 = self.cari_encoders(photo)
            dp0_cari2 = self.cari_encoders(cari2)
            dp0_cari3 = self.cari_encoders(cari3)
            dp0_zr = torch.cat([(dp0_cari0*0.2 + dp0_cari1*0.8 + dp0_cari2*0 + dp0_cari3*0), dp0_photo], dim=1)

        if is_cari:
            dp0_Wact = self.cari_decoders(dp0_zr)
        else:
            dp0_Wact = self.photo_decoders(dp0_zr)

        dp0_Wact = dp0_Wact.permute(0, 2, 3, 1)
        return dp0_Wact



# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGeneratorUnit(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,  num_style=8,
                     padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGeneratorUnit, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        model2 = [nn.Conv2d(num_style, ngf, kernel_size=3, padding=1,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]


        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            model2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        model3 = []
        for i in range(n_blocks):
            model3 += [
                ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i) * 2
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2)//4, x.size(3)//4)
        x = self.model1(x)
        c = self.model2(c)
        input = torch.cat([x, c], dim=1)
        ret = self.model3(input)
        return ret


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_layer != None:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                    norm_layer(dim),
                    nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                    nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if norm_layer != None:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                    norm_layer(dim)]
        else:

            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)







# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, sn = False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        if sn:
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
        else:
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]


        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        if sn:
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        if sn:
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)





# Defines the PatchGAN discriminator with the specified arguments.
class DilatedDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(DilatedDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence1 = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence1 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence1 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence2 = [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence2 += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.LeakyReLU(0.2, True)
        ]
        sequence2 += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.LeakyReLU(0.2, True)
        ]

        sequence3 = [
            nn.Conv2d(ndf * nf_mult * 2, ndf * nf_mult, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True)
        ]
        sequence3 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence3 += [nn.Sigmoid()]

        #self.model = nn.Sequential(*sequence)
        self.layer1 = nn.Sequential(*sequence1)
        self.layer2 = nn.Sequential(*sequence2)
        self.layer3 = nn.Sequential(*sequence3)

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = torch.cat((x1, x2), 1)
        output = self.layer3(x3)
        return output



class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))



class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_nc, image_size=128, conv_dim=64, c_dim=5, repeat_num=4):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.psp = nn.Sequential(PyramidPooling(curr_dim))
        curr_dim = curr_dim * 2
        self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False))

    def forward(self, x):
        h = self.main(x)
        h = self.psp(h)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)
