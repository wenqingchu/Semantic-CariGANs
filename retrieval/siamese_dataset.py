import os.path
from base_dataset import BaseDataset, get_transform, get_target_transform
from image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms
import pdb

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


class SiameseDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, mode='Train'):
        super(SiameseDataset, self).__init__()
        self.mode = mode
        self.train_dir = './data/photo_parse_train/'
        self.test_dir = './data/photo_parse_test/'
        self.cari_dir = './data/caricature_parse/'

        self.train_path = make_dataset(self.train_dir)
        self.train_paths = sorted(self.train_path)
        self.train_dict = {}
        for img_name in self.train_paths:
            person_name = img_name.split('/')[-1]
            person_name = person_name[:-11]
            if person_name not in self.train_dict.keys():
                self.train_dict[person_name] = [img_name]
            else:
                self.train_dict[person_name].append(img_name)
        self.train_size = len(self.train_path)

        self.test_path = make_dataset(self.test_dir)
        self.test_paths = sorted(self.test_path)
        self.test_size = len(self.test_path)

        self.cari_path = make_dataset(self.cari_dir)
        self.cari_paths = sorted(self.cari_path)
        self.cari_dict = {}
        for img_name in self.cari_paths:
            person_name = img_name.split('/')[-1]
            person_name = person_name[:-11]
            if person_name not in self.cari_dict.keys():
                self.cari_dict[person_name] = [img_name]
            else:
                self.cari_dict[person_name].append(img_name)
        self.cari_size = len(self.cari_path)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.target_transform = self.transform
        if self.mode == 'Val':
            self.train_size = self.test_size
            self.train_path = self.test_path
            self.train_paths = self.test_paths

    def __getitem__(self, index):


        index_A = random.randint(0, self.train_size - 1)
        photo_train_path = self.train_paths[index_A]
        person_name = photo_train_path.split('/')[-1]
        person_name = person_name[:-11]
        should_get_same_class = random.randint(0,1)
        if person_name not in self.cari_dict.keys():
            cari_num = 0
        else:
            cari_num = len(self.cari_dict[person_name])
        same_id = 0
        if should_get_same_class and cari_num>0:
            same_id = 1
            cari_number = random.randint(0, cari_num-1)
            cari_path = self.cari_dict[person_name][cari_number]
        else:
            while True:
                index_B = random.randint(0, self.cari_size - 1)
                cari_path = self.cari_paths[index_B]
                cari_name = cari_path.split('/')[-1]
                if cari_name[:-11] != person_name:
                    same_id = 0
                    break



        A = Image.open(photo_train_path)
        A = channel_1toN(A, 10)

        B = Image.open(cari_path)
        B = channel_1toN(B, 10)


        return {'A': A,'B': B, 'label': same_id}

    def __len__(self):
        max_size = 0
        max_size = max(self.train_size, self.cari_size)
        return self.train_size

    def name(self):
        return 'SiameseDataset'
