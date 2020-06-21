import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import random
from .lr_scheduler import LR_Scheduler
from torch.nn import CrossEntropyLoss
import pdb
import torchvision.transforms as transforms
import sys

from models.tps_grid_gen import TPSGridGen
from models.grid_sample import grid_sample
import itertools

class ParserefGANModel(BaseModel):
    def name(self):
        return 'ParserefGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default DeformingGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.palette = np.array([0, 0, 0, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 255, 0, 0, 250, 170, 30,
                   0, 0, 230, 0, 80, 100, 152, 251, 152, 0, 255, 255, 0, 0, 142, 119, 11, 32])
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette = np.append(self.palette, 0)
        self.palette = self.palette.reshape((256, 3))

        # Model configurations.
        self.c_dim = 9
        self.image_size = 64
        self.g_conv_dim = 64
        self.d_conv_dim = 32
        self.g_repeat_num = 5
        self.d_repeat_num = 3
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10

        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.CLSloss = CrossEntropyLoss()
        coord_x = torch.arange(64)
        coord_x = coord_x.repeat(64,1)
        coord_x = coord_x.unsqueeze(0)
        coord_y = torch.arange(64)
        coord_y = coord_y.repeat(64,1)
        coord_y = torch.t(coord_y)
        coord_y = coord_y.unsqueeze(0)
        coord_embedding = torch.cat((coord_x, coord_y), 0)
        coord_embedding = coord_embedding.repeat(opt.batch_size,1,1,1)
        self.coord_embedding = coord_embedding.float().cuda()

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        loss_d_names = ['d_real', 'd_fake', 'd_gp']
        loss_g_names = ['g_fake', 'g_rec', 'g_rec_embedding', 'g_ref']
        loss_grid_names = ['tvw_parse_in', 'tvw_rec', 'br_parse_in', 'br_rec']
        self.loss_names = loss_d_names + loss_g_names + loss_grid_names
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_in = ['parse_ref_color', 'parse_in_color', 'transformed_parse_in_color', 'rec_in_color']
        visual_names_grid = ['parse_in_color_grid','rec_in_color_grid']
        visual_names_in_faces = ['parse_ref_face', 'parse_in_face', 'transformed_parse_in_face', 'rec_in_face']
        visual_names_val_faces = ['parse_ref_color', 'parse_ref_face', 'val_in_color', 'val_in_face', 'transformed_val_in_color', 'transformed_val_in_face']
        blend_visual_names_val_faces = ['parse_ref_A_color', 'parse_ref_A_face', 'parse_ref_B_color', 'parse_ref_B_face', 'val_in_color', 'val_in_face', 'transformed_val_in_color', 'transformed_val_in_face']

        if opt.phase == 'train':
            self.visual_names = visual_names_in + visual_names_grid + visual_names_in_faces
        else:
            self.visual_names = visual_names_val_faces
            if opt.test_phase == 'blend':
                self.visual_names = blend_visual_names_val_faces

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_S(opt.input_nc, opt.output_nc, opt.ngf, 'parseref_gan', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.Discriminator(opt.output_nc, self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.zeroWarp = torch.cuda.FloatTensor(1, 2, 64, 64).fill_(0).to(self.device)
        self.baseg = networks.getBaseGrid(N=64, getbatch=True, batchSize=opt.batch_size).to(self.device)


        # criteria/loss
        self.criterionTVWarp = networks.TotalVaryLoss(opt)
        self.criterionBiasReduce = networks.BiasReduceLoss(opt)

        # initialize optimizers

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

    def set_input(self, input):
        self.image_paths = input['val_path']

        parse_ref = input['A']
        parse_ref_face = input['A_face']


        parse_in = input['B']
        parse_in_face = input['B_face']

        self.parse_ref = parse_ref.to(self.device)
        self.parse_in = parse_in.to(self.device)
        self.parse_in_face = parse_in_face.to(self.device)
        self.parse_ref_face = parse_ref_face.to(self.device)

        val_in = input['val']
        val_in_face = input['val_face']
        self.val_in = val_in.to(self.device)
        self.val_in_face = val_in_face.to(self.device)


        #  visualize the val_in
        val_in_color = input['val'][0].numpy().copy()
        val_in_color = val_in_color.transpose(1, 2, 0)
        val_in_color = np.asarray(np.argmax(val_in_color, axis=2), dtype=np.uint8)
        val_in_color_numpy = np.zeros((val_in_color.shape[0], val_in_color.shape[1], 3))
        for i in range(13):
            val_in_color_numpy[val_in_color == i] = self.palette[i]
        val_in_color = val_in_color_numpy.astype(np.uint8)
        val_in_color = val_in_color.transpose(2, 0, 1)
        val_in_color = val_in_color[np.newaxis, :]
        self.val_in_color = (torch.from_numpy(val_in_color).float()/255.0*2-1).to(self.device)





        #  visualize the parse_ref
        parse_ref_color = input['A'][0].numpy().copy()
        parse_ref_color = parse_ref_color.transpose(1, 2, 0)
        parse_ref_color = np.asarray(np.argmax(parse_ref_color, axis=2), dtype=np.uint8)
        parse_ref_color_numpy = np.zeros((parse_ref_color.shape[0], parse_ref_color.shape[1], 3))
        for i in range(13):
            parse_ref_color_numpy[parse_ref_color == i] = self.palette[i]
        parse_ref_color = parse_ref_color_numpy.astype(np.uint8)
        parse_ref_color = parse_ref_color.transpose(2, 0, 1)
        parse_ref_color = parse_ref_color[np.newaxis, :]
        self.parse_ref_color = (torch.from_numpy(parse_ref_color).float()/255.0*2-1).to(self.device)


        #  visualize the parse_in
        parse_in_color = input['B'][0].numpy().copy()
        parse_in_color = parse_in_color.transpose(1, 2, 0)
        parse_in_color = np.asarray(np.argmax(parse_in_color, axis=2), dtype=np.uint8)
        parse_in_color_numpy = np.zeros((parse_in_color.shape[0], parse_in_color.shape[1], 3))
        for i in range(13):
            parse_in_color_numpy[parse_in_color == i] = self.palette[i]
        parse_in_color = parse_in_color_numpy.astype(np.uint8)
        parse_in_color = parse_in_color.transpose(2, 0, 1)
        parse_in_color = parse_in_color[np.newaxis, :]
        self.parse_in_color = (torch.from_numpy(parse_in_color).float()/255.0*2-1).to(self.device)



    def set_input_val(self, input):
        self.image_paths = input['val_path']

        parse_ref = input['A']
        parse_ref_face = input['A_face']


        self.parse_ref = parse_ref.to(self.device)
        self.parse_ref_face = parse_ref_face.to(self.device)

        val_in = input['val']
        val_in_large = input['val_large']
        val_in_face = input['val_face']
        self.val_in = val_in.to(self.device)
        self.val_in_large = val_in_large.to(self.device)
        self.val_in_face = val_in_face.to(self.device)



        #  visualize the val_in
        val_in_color = input['val'][0].numpy().copy()
        val_in_color = val_in_color.transpose(1, 2, 0)
        val_in_color = np.asarray(np.argmax(val_in_color, axis=2), dtype=np.uint8)
        val_in_color_numpy = np.zeros((val_in_color.shape[0], val_in_color.shape[1], 3))
        for i in range(13):
            val_in_color_numpy[val_in_color == i] = self.palette[i]
        val_in_color = val_in_color_numpy.astype(np.uint8)
        val_in_color = val_in_color.transpose(2, 0, 1)
        val_in_color = val_in_color[np.newaxis, :]
        self.val_in_color = (torch.from_numpy(val_in_color).float()/255.0*2-1).to(self.device)


        #  visualize the val_in
        val_in_large_color = input['val_large'][0].numpy().copy()
        val_in_large_color = val_in_large_color.transpose(1, 2, 0)
        val_in_large_color = np.asarray(np.argmax(val_in_large_color, axis=2), dtype=np.uint8)
        val_in_large_color_numpy = np.zeros((val_in_large_color.shape[0], val_in_large_color.shape[1], 3))
        for i in range(13):
            val_in_large_color_numpy[val_in_large_color == i] = self.palette[i]
        val_in_large_color = val_in_large_color_numpy.astype(np.uint8)
        val_in_large_color = val_in_large_color.transpose(2, 0, 1)
        val_in_large_color = val_in_large_color[np.newaxis, :]
        self.val_in_large_color = (torch.from_numpy(val_in_large_color).float()/255.0*2-1).to(self.device)





        #  visualize the parse_ref
        parse_ref_color = input['A'][0].numpy().copy()
        parse_ref_color = parse_ref_color.transpose(1, 2, 0)
        parse_ref_color = np.asarray(np.argmax(parse_ref_color, axis=2), dtype=np.uint8)
        parse_ref_color_numpy = np.zeros((parse_ref_color.shape[0], parse_ref_color.shape[1], 3))
        for i in range(13):
            parse_ref_color_numpy[parse_ref_color == i] = self.palette[i]
        parse_ref_color = parse_ref_color_numpy.astype(np.uint8)
        parse_ref_color = parse_ref_color.transpose(2, 0, 1)
        parse_ref_color = parse_ref_color[np.newaxis, :]
        self.parse_ref_color = (torch.from_numpy(parse_ref_color).float()/255.0*2-1).to(self.device)



    def set_input_blend(self, input):
        self.image_paths = input['val_path']

        parse_ref_A = input['A']
        parse_ref_A_face = input['A_face']

        parse_ref_B = input['B']
        parse_ref_B_face = input['B_face']

        parse_ref_C = input['C']
        parse_ref_C_face = input['C_face']


        self.parse_ref_A = parse_ref_A.to(self.device)
        self.parse_ref_B = parse_ref_B.to(self.device)
        self.parse_ref_C = parse_ref_C.to(self.device)
        self.parse_ref_A_face = parse_ref_A_face.to(self.device)
        self.parse_ref_B_face = parse_ref_B_face.to(self.device)
        self.parse_ref_C_face = parse_ref_C_face.to(self.device)

        val_in = input['val']
        val_in_face = input['val_face']
        self.val_in = val_in.to(self.device)
        self.val_in_face = val_in_face.to(self.device)



        #  visualize the val_in
        val_in_color = input['val'][0].numpy().copy()
        val_in_color = val_in_color.transpose(1, 2, 0)
        val_in_color = np.asarray(np.argmax(val_in_color, axis=2), dtype=np.uint8)
        val_in_color_numpy = np.zeros((val_in_color.shape[0], val_in_color.shape[1], 3))
        for i in range(13):
            val_in_color_numpy[val_in_color == i] = self.palette[i]
        val_in_color = val_in_color_numpy.astype(np.uint8)
        val_in_color = val_in_color.transpose(2, 0, 1)
        val_in_color = val_in_color[np.newaxis, :]
        self.val_in_color = (torch.from_numpy(val_in_color).float()/255.0*2-1).to(self.device)





        #  visualize the parse_ref_A
        parse_ref_A_color = input['A'][0].numpy().copy()
        parse_ref_A_color = parse_ref_A_color.transpose(1, 2, 0)
        parse_ref_A_color = np.asarray(np.argmax(parse_ref_A_color, axis=2), dtype=np.uint8)
        parse_ref_A_color_numpy = np.zeros((parse_ref_A_color.shape[0], parse_ref_A_color.shape[1], 3))
        for i in range(13):
            parse_ref_A_color_numpy[parse_ref_A_color == i] = self.palette[i]
        parse_ref_A_color = parse_ref_A_color_numpy.astype(np.uint8)
        parse_ref_A_color = parse_ref_A_color.transpose(2, 0, 1)
        parse_ref_A_color = parse_ref_A_color[np.newaxis, :]
        self.parse_ref_A_color = (torch.from_numpy(parse_ref_A_color).float()/255.0*2-1).to(self.device)

        #  visualize the parse_ref_B
        parse_ref_B_color = input['B'][0].numpy().copy()
        parse_ref_B_color = parse_ref_B_color.transpose(1, 2, 0)
        parse_ref_B_color = np.asarray(np.argmax(parse_ref_B_color, axis=2), dtype=np.uint8)
        parse_ref_B_color_numpy = np.zeros((parse_ref_B_color.shape[0], parse_ref_B_color.shape[1], 3))
        for i in range(13):
            parse_ref_B_color_numpy[parse_ref_B_color == i] = self.palette[i]
        parse_ref_B_color = parse_ref_B_color_numpy.astype(np.uint8)
        parse_ref_B_color = parse_ref_B_color.transpose(2, 0, 1)
        parse_ref_B_color = parse_ref_B_color[np.newaxis, :]
        self.parse_ref_B_color = (torch.from_numpy(parse_ref_B_color).float()/255.0*2-1).to(self.device)


    def backward_D(self):
        # Compute loss with real images.
        out_src, out_cls = self.netD(self.parse_ref)
        d_loss_real = - torch.mean(out_src)

        transformed_parse_in_Wact = self.netG(self.parse_in, self.parse_ref)
        x_fake = F.grid_sample(self.parse_in, transformed_parse_in_Wact, padding_mode='border')

        out_src, out_cls = self.netD(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(self.parse_ref.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * self.parse_ref.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.netD(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.optimizer_D.step()
        # Logging.
        self.loss_d_real = d_loss_real
        self.loss_d_fake = d_loss_fake
        self.loss_d_gp = d_loss_gp


    def backward_G(self):
        batch_size = self.parse_in.size(0)
        coord_embedding = self.coord_embedding[:batch_size]

        transformed_parse_in_Wact = self.netG(self.parse_in, self.parse_ref)
        transformed_parse_in = F.grid_sample(self.parse_in, transformed_parse_in_Wact, padding_mode='border')
        transformed_parse_in_coord_embedding = F.grid_sample(coord_embedding, transformed_parse_in_Wact,                                                                  padding_mode='border')
        parse_in_face_Wact = F.upsample(transformed_parse_in_Wact.permute(0, 3, 1, 2), size=(256, 256),
                                 mode='bilinear')
        transformed_parse_in_face = F.grid_sample(self.parse_in_face, parse_in_face_Wact.permute(0, 2, 3, 1))

        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2,3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face


        out_src, out_cls = self.netD(transformed_parse_in)
        g_loss_fake = - torch.mean(out_src)
        rec_in_Wact = self.netG(transformed_parse_in, self.parse_in)
        rec_in = F.grid_sample(transformed_parse_in, rec_in_Wact, padding_mode='border')
        rec_coord_embedding = F.grid_sample(transformed_parse_in_coord_embedding,
                                                       rec_in_Wact, padding_mode='border')

        g_loss_rec = torch.mean(torch.abs(self.parse_in - rec_in))
        g_loss_rec_embedding = torch.mean(torch.abs(coord_embedding - rec_coord_embedding))
        rec_face_Wact = F.upsample(rec_in_Wact.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')
        rec_face = F.grid_sample(transformed_parse_in_face, rec_face_Wact.permute(0, 2, 3, 1))

        back_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3))
        skin_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3))
        #component_eye_weight = torch.ones(self.parse_ref.size(0), 4, self.parse_ref.size(2), self.parse_ref.size(3)) * 20
        component_eye_weight = torch.ones(self.parse_ref.size(0), 4, self.parse_ref.size(2), self.parse_ref.size(3)) * 10
        component_nose_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3)) * 4
        component_up_mouth_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3)) * 10
        component_middle_mouth_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3)) * 2
        component_down_mouth_weight = torch.ones(self.parse_ref.size(0), 1, self.parse_ref.size(2), self.parse_ref.size(3)) * 10
        face_weight = torch.cat((back_weight, skin_weight, component_eye_weight, component_nose_weight, component_up_mouth_weight, component_middle_mouth_weight, component_down_mouth_weight), 1).cuda()
        g_loss_ref_rec = torch.mean(face_weight * torch.abs(self.parse_ref - transformed_parse_in))
        parse_ref_sum = F.avg_pool2d(face_weight * self.parse_ref, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))

        transformed_parse_in_sum = F.avg_pool2d(face_weight * transformed_parse_in, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))
        g_loss_ref_sum = torch.mean(torch.abs(parse_ref_sum - transformed_parse_in_sum))
        g_loss_ref_rec = g_loss_ref_rec + 2 * g_loss_ref_sum



        tmp_x = torch.arange(64) / 64.0
        tmp_x = tmp_x.repeat(64,1)
        tmp_x = tmp_x.unsqueeze(0)
        pixel_location_x = tmp_x.repeat(self.parse_ref.size(0),1,1,1).cuda().float()
        tmp_y = torch.arange(64) / 64.0
        tmp_y = tmp_y.repeat(64,1)
        tmp_y = torch.t(tmp_y)
        tmp_y = tmp_y.unsqueeze(0)
        pixel_location_y = tmp_y.repeat(self.parse_ref.size(0),1,1,1).cuda().float()

        parse_ref_loc_x = F.avg_pool2d(face_weight * pixel_location_x * self.parse_ref, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))
        parse_ref_loc_y = F.avg_pool2d(face_weight * pixel_location_y * self.parse_ref, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))
        transformed_parse_in_loc_x = F.avg_pool2d(face_weight * pixel_location_x * transformed_parse_in, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))
        transformed_parse_in_loc_y = F.avg_pool2d(face_weight * pixel_location_y * transformed_parse_in, kernel_size=(self.parse_ref.size(2), self.parse_ref.size(3)))
        g_loss_ref_loc = torch.mean(torch.abs(parse_ref_loc_x - transformed_parse_in_loc_x))
        g_loss_ref_loc = g_loss_ref_loc + torch.mean(torch.abs(parse_ref_loc_y - transformed_parse_in_loc_y))
        g_loss_ref_rec = g_loss_ref_rec + 2 * g_loss_ref_loc






        tvw_weight = 1e-3
        loss_tvw_parse_in = self.criterionTVWarp(transformed_parse_in_Wact.permute(0, 3, 1, 2) - self.baseg[:batch_size], weight=tvw_weight)
        loss_tvw_rec = self.criterionTVWarp(rec_in_Wact.permute(0, 3, 1, 2) - self.baseg[:batch_size], weight=tvw_weight)
        loss_br_parse_in = self.criterionBiasReduce(transformed_parse_in_Wact.permute(0, 3, 1, 2) - self.baseg[:batch_size],
            self.zeroWarp[:batch_size], weight=1)
        loss_br_rec = self.criterionBiasReduce(
            rec_in_Wact.permute(0, 3, 1, 2) - self.baseg[:batch_size], self.zeroWarp[:batch_size], weight=1)

        # all loss functions

        # Backward and optimize.
        self.g_loss_fake = g_loss_fake * 1
        self.g_loss_rec = g_loss_rec * 2
        self.g_loss_rec_embedding = g_loss_rec_embedding * 2
        self.g_loss_ref = g_loss_ref_rec * 500
        self.loss_tvw_parse_in = loss_tvw_parse_in * 0.1
        self.loss_tvw_rec = loss_tvw_rec * 0.1
        self.loss_br_parse_in = loss_br_parse_in
        self.loss_br_rec = loss_br_rec


        g_loss = self.g_loss_fake + self.lambda_rec * self.g_loss_rec + self.lambda_rec * self.g_loss_rec_embedding\
                     + self.loss_tvw_parse_in + self.loss_tvw_rec + self.loss_br_parse_in + self.loss_br_rec + self.g_loss_ref

        self.reset_grad()
        g_loss.backward()
        self.optimizer_G.step()
        # Logging.
        self.loss_g_fake = self.g_loss_fake
        self.loss_g_rec = self.g_loss_rec
        self.loss_g_rec_embedding = self.g_loss_rec_embedding
        self.loss_g_ref = self.g_loss_ref


        source_control_points = transformed_parse_in_Wact[0].cpu().float().detach()
        parse_in_color = self.parse_in_color[0].cpu().float().numpy()
        parse_in_color = (parse_in_color + 1.0) / 2 * 255
        parse_in_color = Image.fromarray(parse_in_color.transpose(1, 2, 0).astype(np.uint8)).convert('RGB').resize((128, 128))
        canvas = Image.new(mode='RGB', size=(64 * 4, 64 * 4), color=(128, 128, 128))
        canvas.paste(parse_in_color, (64, 64))
        source_points = (source_control_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)

        grid_size = 8
        grid_step = int(64 / 8)
        for j in range(grid_size):
            for k in range(grid_size):
                x, y = source_points[j*grid_step + 4][k*grid_step + 4]
                draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

        source_points = source_points.view(64, 64, 2)
        for j in range(grid_size):
            for k in range(grid_size):
                x1, y1 = source_points[j*grid_step + 4, k*grid_step + 4]
                if j > 0:  # connect to left
                    x2, y2 = source_points[(j - 1)*grid_step + 4, k*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j*grid_step + 4, (k - 1)*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        parse_in_color = np.asarray(canvas, np.uint8)
        parse_in_color_grid = parse_in_color.transpose(2, 0, 1)
        parse_in_color_grid = parse_in_color_grid[np.newaxis, :]
        self.parse_in_color_grid = (torch.from_numpy(parse_in_color_grid).float()/255*2-1).to(self.device)


        # visualize the fake_B
        transformed_parse_in_color = transformed_parse_in.data[0].cpu().numpy()
        transformed_parse_in_color = transformed_parse_in_color.transpose(1,2,0)
        transformed_parse_in_color = np.asarray(np.argmax(transformed_parse_in_color, axis=2), dtype=np.uint8)
        transformed_parse_in_color_numpy = np.zeros((transformed_parse_in_color.shape[0], transformed_parse_in_color.shape[1],3))
        for i in range(13):
            transformed_parse_in_color_numpy[transformed_parse_in_color==i] = self.palette[i]
        transformed_parse_in_color = transformed_parse_in_color_numpy.astype(np.uint8)
        transformed_parse_in_color = transformed_parse_in_color.transpose(2, 0, 1)
        transformed_parse_in_color = transformed_parse_in_color[np.newaxis, :]
        self.transformed_parse_in_color = (torch.from_numpy(transformed_parse_in_color).float()/255*2-1).to(self.device)

        self.transformed_parse_in_face = transformed_parse_in_face




        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        transformed_val_in_color_numpy = np.zeros((transformed_val_in_color.shape[0], transformed_val_in_color.shape[1],3))
        for i in range(13):
            transformed_val_in_color_numpy[transformed_val_in_color==i] = self.palette[i]
        transformed_val_in_color = transformed_val_in_color_numpy.astype(np.uint8)
        transformed_val_in_color = transformed_val_in_color.transpose(2, 0, 1)
        transformed_val_in_color = transformed_val_in_color[np.newaxis, :]
        self.transformed_val_in_color = (torch.from_numpy(transformed_val_in_color).float()/255*2-1).to(self.device)



        # visualize the rec_A

        source_control_points = rec_in_Wact[0].cpu().float().detach()
        real_A_color = self.transformed_parse_in_color[0].cpu().float().numpy()
        real_A_color = (real_A_color + 1.0) / 2 * 255
        real_A_color = Image.fromarray(real_A_color.transpose(1, 2, 0).astype(np.uint8)).convert('RGB').resize((128, 128))
        canvas = Image.new(mode='RGB', size=(64 * 4, 64 * 4), color=(128, 128, 128))
        canvas.paste(real_A_color, (64, 64))
        source_points = (source_control_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)

        for j in range(grid_size):
            for k in range(grid_size):
                x, y = source_points[j*grid_step + 4][k*grid_step + 4]
                draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

        source_points = source_points.view(64, 64, 2)
        for j in range(grid_size):
            for k in range(grid_size):
                x1, y1 = source_points[j*grid_step + 4, k*grid_step + 4]
                if j > 0:  # connect to left
                    x2, y2 = source_points[(j - 1)*grid_step + 4, k*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j*grid_step + 4, (k - 1)*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        real_A_color = np.asarray(canvas, np.uint8)
        real_A_color_grid = real_A_color.transpose(2, 0, 1)
        real_A_color_grid = real_A_color_grid[np.newaxis, :]
        self.rec_in_color_grid = (torch.from_numpy(real_A_color_grid).float()/255*2-1).to(self.device)



        # visualize the rec
        rec_in_color = rec_in.data[0].cpu().numpy()
        rec_in_color = rec_in_color.transpose(1,2,0)
        rec_in_color = np.asarray(np.argmax(rec_in_color, axis=2), dtype=np.uint8)
        rec_in_color_numpy = np.zeros((rec_in_color.shape[0], rec_in_color.shape[1],3))
        for i in range(13):
            rec_in_color_numpy[rec_in_color==i] = self.palette[i]
        rec_in_color = rec_in_color_numpy.astype(np.uint8)
        rec_in_color = rec_in_color.transpose(2, 0, 1)
        rec_in_color = rec_in_color[np.newaxis, :]
        self.rec_in_color = (torch.from_numpy(rec_in_color).float()/255*2-1).to(self.device)
        self.rec_in_face = rec_face


    def print_loss(self):
        print(self.loss_g_ref.item(), self.loss_g_rec.item(), self.loss_g_rec_embedding.item(), self.loss_g_fake.item())


    def optimize_parameters(self):
        self.backward_D()
        self.backward_G()



    def forward(self):
        #self.netG.eval()
        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        transformed_val_in_color_numpy = np.zeros((transformed_val_in_color.shape[0], transformed_val_in_color.shape[1],3))
        for i in range(13):
            transformed_val_in_color_numpy[transformed_val_in_color==i] = self.palette[i]
        transformed_val_in_color = transformed_val_in_color_numpy.astype(np.uint8)
        transformed_val_in_color = transformed_val_in_color.transpose(2, 0, 1)
        transformed_val_in_color = transformed_val_in_color[np.newaxis, :]
        self.transformed_val_in_color = (torch.from_numpy(transformed_val_in_color).float()/255*2-1).to(self.device)


    def test_single(self):
        self.netG.eval()
        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        transformed_val_in_color_numpy = np.zeros((transformed_val_in_color.shape[0], transformed_val_in_color.shape[1],3))
        for i in range(13):
            transformed_val_in_color_numpy[transformed_val_in_color==i] = self.palette[i]
        transformed_val_in_color = transformed_val_in_color_numpy.astype(np.uint8)
        transformed_val_in_color = transformed_val_in_color.transpose(2, 0, 1)
        transformed_val_in_color = transformed_val_in_color[np.newaxis, :]
        self.transformed_val_in_color = (torch.from_numpy(transformed_val_in_color).float()/255*2-1).to(self.device)



    def test_val(self):
        self.netG.eval()
        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        transformed_val_in = F.grid_sample(self.val_in_large, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        return transformed_val_in_color, transformed_val_in_face

    def test_single_control(self):
        self.netG.eval()
        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        transformed_val_in = F.grid_sample(self.val_in_large, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        source_control_points = transformed_val_in_Wact[0].cpu().float().detach()
        parse_in_color = self.val_in_large_color[0].cpu().float().numpy()
        parse_in_color = (parse_in_color + 1.0) / 2 * 255
        parse_in_color = Image.fromarray(parse_in_color.transpose(1, 2, 0).astype(np.uint8)).convert('RGB').resize((256, 256))
        canvas = Image.new(mode='RGB', size=(64 * 4, 64 * 4), color=(128, 128, 128))
        canvas.paste(parse_in_color, (0, 0))
        source_points = (source_control_points + 1) / 2 * 256
        draw = ImageDraw.Draw(canvas)
        grid_size = 8
        grid_step = int(64 / 8)
        for j in range(grid_size):
            for k in range(grid_size):
                x, y = source_points[j*grid_step + 4][k*grid_step + 4]
                draw.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))
        source_points = source_points.view(64, 64, 2)
        for j in range(grid_size):
            for k in range(grid_size):
                x1, y1 = source_points[j*grid_step + 4, k*grid_step + 4]
                if j > 0:  # connect to left
                    x2, y2 = source_points[(j - 1)*grid_step + 4, k*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                if k > 0:  # connect to up
                    x2, y2 = source_points[j*grid_step + 4, (k - 1)*grid_step + 4]
                    draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
        parse_in_color_grid = np.asarray(canvas, np.uint8)
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        return transformed_val_in_color, parse_in_color_grid, transformed_val_in_face




    def generate_feature(self):
        self.netG.eval()
        transformed_val_in_Wact, cari_feature = self.netG(self.val_in, self.parse_ref)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        transformed_val_in_color_numpy = np.zeros((transformed_val_in_color.shape[0], transformed_val_in_color.shape[1],3))
        for i in range(13):
            transformed_val_in_color_numpy[transformed_val_in_color==i] = self.palette[i]
        transformed_val_in_color = transformed_val_in_color_numpy.astype(np.uint8)
        transformed_val_in_color = transformed_val_in_color.transpose(2, 0, 1)
        transformed_val_in_color = transformed_val_in_color[np.newaxis, :]
        self.transformed_val_in_color = (torch.from_numpy(transformed_val_in_color).float()/255*2-1).to(self.device)
        return cari_feature





    def test_blend(self):
        self.netG.eval()
        transformed_val_in_Wact = self.netG(self.val_in, self.parse_ref_A, True, True, self.parse_ref_B, self.parse_ref_C)
        transformed_val_in = F.grid_sample(self.val_in, transformed_val_in_Wact,padding_mode='border')
        val_in_face_Wact = F.upsample(transformed_val_in_Wact.permute(0, 3, 1, 2), size=(256,256), mode = 'bilinear')
        transformed_val_in_face = F.grid_sample(self.val_in_face, val_in_face_Wact.permute(0, 2, 3, 1))
        self.transformed_val_in = transformed_val_in
        self.transformed_val_in_face = transformed_val_in_face

        # visualize the val fake_B
        transformed_val_in_color = transformed_val_in.data[0].cpu().numpy()
        transformed_val_in_color = transformed_val_in_color.transpose(1,2,0)
        transformed_val_in_color = np.asarray(np.argmax(transformed_val_in_color, axis=2), dtype=np.uint8)
        transformed_val_in_color_numpy = np.zeros((transformed_val_in_color.shape[0], transformed_val_in_color.shape[1],3))
        for i in range(13):
            transformed_val_in_color_numpy[transformed_val_in_color==i] = self.palette[i]
        transformed_val_in_color = transformed_val_in_color_numpy.astype(np.uint8)
        transformed_val_in_color = transformed_val_in_color.transpose(2, 0, 1)
        transformed_val_in_color = transformed_val_in_color[np.newaxis, :]
        self.transformed_val_in_color = (torch.from_numpy(transformed_val_in_color).float()/255*2-1).to(self.device)


