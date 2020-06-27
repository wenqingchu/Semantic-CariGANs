import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
from data.base_dataset import BaseDataset, get_transform, get_target_transform
import numpy as np
from parsing.psp import PSP
from retrieval.siamese import SiameseNetwork
import AdaIn.net as StyleNet
from AdaIn.function import adaptive_instance_normalization
from torch.nn import functional as F
import pdb

palette = np.array([0, 0, 0, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 255, 0, 0, 250, 170, 30,
                         0, 0, 230, 0, 80, 100, 152, 251, 152, 0, 255, 255, 0, 0, 142, 119, 11, 32])
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette = np.append(palette, 0)


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0,interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f,style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

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

def parsing_img(parsing_net, img):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    resize = nn.Upsample(size=(256,256), mode='bilinear',align_corners=True)
    with torch.no_grad():
        image_384 = img.resize((384, 384), Image.BILINEAR)
        img_384 = to_tensor(image_384)
        img_384 = torch.unsqueeze(img_384, 0)
        img_384 = img_384.cuda()
        out_384, _ = parsing_net(img_384)
        out_384 = resize(out_384)

        image_flip = image_384.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip = to_tensor(image_flip)
        img_flip = torch.unsqueeze(img_flip, 0)
        img_flip = img_flip.cuda()
        out_flip, _ = parsing_net(img_flip)
        out_flip_384 = resize(out_flip)

        image_256 = img.resize((256, 256), Image.BILINEAR)
        img_256 = to_tensor(image_256)
        img_256 = torch.unsqueeze(img_256, 0)
        img_256 = img_256.cuda()
        out_256, _ = parsing_net(img_256)
        out = (out_384 + out_256) / 4

        image_flip = image_256.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip = to_tensor(image_flip)
        img_flip = torch.unsqueeze(img_flip, 0)
        img_flip = img_flip.cuda()
        out_flip, _ = parsing_net(img_flip)
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

        return parsing



if __name__ == '__main__':
    # loading model

    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.test_phase = 'single'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    model.eval()



    print('Parse the input image!')
    n_classes = 11
    parsing_net = PSP(n_classes, 'resnet50')
    parsing_net.load_state_dict(torch.load(opt.parsing_model))
    parsing_net.cuda()
    parsing_net.eval()

    transform = get_transform(opt)

    photo_face_name = opt.input

    sample_face_path = photo_face_name
    sample_face = Image.open(sample_face_path).convert('RGB')


    sample_mask = parsing_img(parsing_net, sample_face)

    sample_mask = sample_mask.astype(np.uint8)
    sample_mask_large = sample_mask.copy()

    sample_mask = Image.fromarray(sample_mask)
    sample_mask = np.array(sample_mask.resize((64, 64), Image.NEAREST))
    sample_mask = channel_1toN(sample_mask, opt.output_nc)
    sample_face = transform(sample_face)
    sample_mask_large = channel_1toN(sample_mask_large, opt.output_nc)



    print('Parsing is done!')
    
    if opt.shape == None:
        #retrieval
        print('Load the caricature gallery!')
        retrieval_net = SiameseNetwork()
        retrieval_net.load_state_dict(torch.load(opt.retrieval_model))
        retrieval_net.cuda()
        retrieval_net.eval()
        cari_name_gallery = []
        cari_f = open('examples/cari_gallery/cari.txt')
        cari_enc= np.zeros((len(cari_f.readlines()), 128))
        cari_f = open('examples/cari_gallery/cari.txt')
        for i, cari_name in enumerate(cari_f.readlines()):
            cari_name = cari_name.strip()
            img0 = Image.open('examples/cari_gallery/parsing_maps_gallery/' + cari_name)
            img0 = img0.resize((256,256), Image.NEAREST)
            img0 = channel_1toN(img0, 10)

            img0 = img0.cuda()
            z1,output1 = retrieval_net.forward_cari(img0.unsqueeze(0))
            cari_enc[i] = z1[0].detach().cpu().numpy()
            cari_name_gallery.append(cari_name)
    
        img0 = sample_mask_large
        img0 = img0.cuda()
        z1,output1 = retrieval_net.forward_photo(img0.unsqueeze(0))
        photo_enc = z1[0].detach().cpu().numpy()
        dist = []
        for j in range(len(cari_name_gallery)):
            dd = np.sqrt(np.sum(np.square(photo_enc - cari_enc[j])))
            dist.append((cari_name_gallery[j],dd))
        dist.sort(key=takeSecond)
        cari_names = []
        for j in range(5):
            cari_names.append(dist[j][0])
        print('The retrieval is done!')
    else:
        cari_names = []
        cari_names.append(opt.shape.split('/')[-1])

    #load style transfer network

    
    vgg = StyleNet.vgg
    style_decoder = StyleNet.decoder
    style_decoder.eval()
    vgg.eval()

    style_decoder.load_state_dict(torch.load(opt.style_decoder_model))
    vgg.load_state_dict(torch.load(opt.style_encoder_model))
    style_encoder = nn.Sequential(*list(vgg.children())[:31])

    style_encoder.cuda()
    style_decoder.cuda()
    content_size = 512
    style_size = 512
    crop = False
    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    #cari_names = ['Scarlett_Johansson_C00001.png', 'Condoleezza_Rice_C00001.png', 'Amanda_Seyfried_C00001.png']

    # shape transformation
    for cari_name in cari_names:
        if opt.shape == None:
            A_path = 'examples/cari_gallery/parsing_maps_gallery/' + cari_name
        else:
            A_path = opt.shape
        if os.path.exists(A_path) == False:
            continue
        #A_face_path = A_path.replace('.png', '.jpg')
        #A_face_path = A_face_path.replace('label', 'image')
        A_face_path = A_path

        A = Image.open(A_path)
        A = A.resize((256,256), Image.NEAREST)

        A = A.resize((64, 64), Image.NEAREST)
        A = channel_1toN(A, opt.output_nc)
        A_img_face = Image.open(A_face_path).convert('RGB')
        A_face = transform(A_img_face)

        data = {'A': A.unsqueeze(0),'val_large': sample_mask_large.unsqueeze(0), 'A_face': A_face.unsqueeze(0), 'val': sample_mask.unsqueeze(0), 'val_face': sample_face.unsqueeze(0), 'val_path': sample_face_path}
        model.set_input_val(data)
        transformed_val_in_color, transformed_val_in_face = model.test_val()
        save_name = photo_face_name.strip()[:-4].split('/')[-1] + '_' + cari_name.strip().replace('png', 'jpg')
        transformed_val_in_color = Image.fromarray(transformed_val_in_color.astype(np.uint8)).convert('P')
        transformed_val_in_color.putpalette(palette.tolist())
        #transformed_val_in_color.save('result_mask_transform/' + save_name)

        transformed_val_in_face = transformed_val_in_face.detach()[0].cpu().float().numpy()
        transformed_val_in_face = (np.transpose(transformed_val_in_face, (1, 2, 0)) + 1) / 2.0 * 255.0
        transformed_val_in_face = transformed_val_in_face.astype(np.uint8)
        transformed_val_in_face = Image.fromarray(transformed_val_in_face)
        #transformed_val_in_face.save('result_face_transform/' + save_name.replace('png','jpg'))

        #style transfer
        #style_dir = 'AdaIn/style_imgs/'
        #style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
        content = content_tf(transformed_val_in_face)
        content = content.cuda().unsqueeze(0)
        style = style_tf(Image.open(opt.style_path).convert('RGB'))
        alpha = 1.0
        style = style.cuda().unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(style_encoder, style_decoder, content, style, alpha)
        output = output.cpu()
        output = F.upsample(output, size=(256,256))
        save_image(output, 'results/' +save_name)


