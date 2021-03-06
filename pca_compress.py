#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 12:45:17 2018

@author: eric
"""
from sklearn.decomposition import PCA
from load_images import load_images
import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
import torch
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

class PCA_Compress():

    def compress(self, X, dims):
        self.pca = PCA(dims)
        lower_dim_data = self.pca.fit_transform(X)
        return lower_dim_data


    def uncompress(self, X_comp):
        approx = self.pca.inverse_transform(X_comp)
        return approx
