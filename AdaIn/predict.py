import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization
from function import coral
import torch.nn.functional as F

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)



decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()
decoder_path = 'models/decoder_iter_320000.pth.tar'
decoder.load_state_dict(torch.load(decoder_path))
encoder_path = 'models/vgg_normalised.pth'
vgg.load_state_dict(torch.load(encoder_path))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.cuda()
decoder.cuda()
content_size = 256
style_size = 256
crop = False
content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

content_path = 'content/P00012.jpg'
style_path = 'style_imgs/Angela_Merkel_C00012.jpg'

# process one content and one style
content = content_tf(Image.open(content_path).convert('RGB'))
style = style_tf(Image.open(style_path).convert('RGB'))
alpha = 1.0
style = style.cuda().unsqueeze(0)
content = content.cuda().unsqueeze(0)
with torch.no_grad():
    output = style_transfer(vgg, decoder, content, style, alpha)

output = output.cpu()
output_name = 'tmp.jpg'
save_image(output, output_name)
