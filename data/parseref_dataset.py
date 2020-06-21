import os.path
from data.base_dataset import BaseDataset, get_transform, get_target_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms

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


class ParserefDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_val = os.path.join(opt.dataroot, 'val')
        self.val_path = make_dataset(self.dir_val)
        self.val_paths = sorted(self.val_path)
        self.val_size = len(self.val_path)

        self.A_path = make_dataset(os.path.join(opt.dataroot, opt.phase + 'A'))
        self.A_path = sorted(self.A_path)
        self.A_size = len(self.A_path)

        self.B_path = make_dataset(os.path.join(opt.dataroot, opt.phase + 'B'))
        self.B_path = sorted(self.B_path)
        self.B_size = len(self.B_path)


        self.transform = get_transform(opt)
        self.target_transform = get_target_transform(opt)

    def __getitem__(self, index):


        if self.opt.serial_batches:
            index_A = index % self.A_size
        else:
            index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_path[index_A]
        A_face_path =A_path.replace('.png', '.jpg')
        A_face_path = A_face_path.replace('train', 'face')




        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_path[index_B]
        B_face_path =B_path.replace('.png', '.jpg')
        B_face_path = B_face_path.replace('train', 'face')



        if self.opt.serial_batches:
            index_val = index % self.val_size
        else:
            index_val = random.randint(0, self.val_size - 1)
        val_path = self.val_paths[index_val]
        val_path_face = val_path.replace('.png', '.jpg')
        val_path_face = val_path_face.replace('val', 'face')


        A_img_face = Image.open(A_face_path).convert('RGB')
        val_img_face = Image.open(val_path_face).convert('RGB')

        A = Image.open(A_path)
        A = A.resize((64 , 64), Image.NEAREST)
        A = channel_1toN(A, self.opt.output_nc)
        A_face = self.transform(A_img_face)

        B = Image.open(B_path)
        B = B.resize((64, 64), Image.NEAREST)
        B = channel_1toN(B, self.opt.output_nc)
        B_img_face = Image.open(B_face_path).convert('RGB')
        B_face = self.transform(B_img_face)

        val = Image.open(val_path)
        val = val.resize((64 , 64), Image.NEAREST)
        val = channel_1toN(val, self.opt.output_nc)
        val_face = self.transform(val_img_face)
        val_label = 0

        return {'A': A, 'A_face': A_face, 'B': B, 'B_face': B_face, 'val': val, 'val_face': val_face, 'val_label': val_label, 'A_path': A_path, 'val_path': val_path}

    def __len__(self):
        max_size = 0
        max_size = max(self.B_size, self.A_size)
        return max_size

    def name(self):
        return 'ParserefDataset'
