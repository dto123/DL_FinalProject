#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:02:11 2018

@author: eric
"""


import os
import matplotlib.pyplot as plt
import numpy as np
import Image


def load_images(max_num_load, h_dim, w_dim):
    imgs = []
    for filename in os.listdir('ImageNet'):
        if filename.endswith(".JPEG"):

            # load image
            jpgfile = Image.open('ImageNet/'+filename)
            s = np.array(jpgfile)
            if(s.shape[0] < h_dim or s.shape[1] < w_dim or s.ndim!=3): continue

            # crop image
            s = s[:h_dim, :w_dim, :]

            imgs.append(s)
            if len(imgs) >= max_num_load: break
            if len(imgs) % 10 == 0: print(len(imgs))

    return np.array(imgs)

imgs=load_images(100,100,100)
plt.imshow(imgs[4])
plt.show()

print(imgs.shape)
