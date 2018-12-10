#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:02:11 2018

@author: eric
"""

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


def load_images(max_num_load, h_dim, w_dim, bw=False):
    imgs = []
    for filename in os.listdir('ImageNet'):
        if filename.endswith(".JPEG"):

            if(bw):
                jpgfile = Image.open('ImageNet/'+filename)
                s = np.array(jpgfile)
                if(s.shape[0] < h_dim or s.shape[1] < w_dim or s.ndim!=2):continue
                # crop image
                s = s[:h_dim, :w_dim]

                imgs.append(s)
                if len(imgs) >= max_num_load: break
                if len(imgs) % 10 == 0: print(len(imgs))
                continue

            # load image
            jpgfile = Image.open('ImageNet/'+filename)
            s = np.array(jpgfile)
            if(s.shape[0] < h_dim or s.shape[1] < w_dim or s.ndim!=3): continue

            #print('original_image: ' ,s.shape)
            #print(h_dim)
            #print(w_dim)

            # crop image
            s = s[:h_dim, :w_dim, :]

            #print('cropped image: ' ,s.shape)

            imgs.append(s)
            if len(imgs) >= max_num_load: break
            if len(imgs) % 10 == 0: print(len(imgs))

    #print('size: ' ,np.array(imgs).shape)
    return np.array(imgs)

#imgs=load_images(1000,500,500)
