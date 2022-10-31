import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images')
    parser.add_argument('--output-path', type=str, help='path of output images')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

   # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im1 = Image.open(os.path.join(args.input_path, im_name))

        # unify image channels to 3
        im1 = np.asarray(im1)
        if len(im1.shape) == 2:
            im1 = im1[:, :, None]
        if im1.shape[2] == 1:
            im1 = np.repeat(im1, 3, axis=2)
        elif im1.shape[2] == 4:
            im1 = im1[:, :, 0:3]
        #im1 array  
        # convert image to PyTorch tensor
        im1 = Image.fromarray(im1)
        im = im_transform(im1)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        
        #im_rh = 512
        #im_rw = 348
        #im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        im = F.adaptive_avg_pool2d(im, (im_rh, im_rw))

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        #matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        # make change so that the foreground output img is  512*348 the needed size in the virtual try-on 
        im_rh = 512
        im_rw = 348
        matte = F.adaptive_avg_pool2d(matte, (im_rh, im_rw))
