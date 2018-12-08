#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:31:06 2018

@author: eric
"""

import random
import numpy as np
from cs682.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import matplotlib.image as image
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.utils.data as Data
from load_images import load_images

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
    
def check_image(model, X, ind, height, width):
    reconstructed = model(X.transpose(1,3)).transpose(1,3)
    reconstructed.clamp_(0,255)
    print(reconstructed.shape)
    img = reconstructed.detach().numpy()[ind].reshape(height, width,3)
    orig_img = X.detach().numpy()[ind].reshape(height, width,3)
    plt.imshow(img.astype('uint8'))    
    plt.show()
    plt2.imshow(orig_img.astype('uint8'))    
    plt2.show()
    return img, orig_img


height, width = 50,50
num_samples = 100

X_train = torch.from_numpy(load_images(num_samples,height,width)).float()


train_loader = Data.DataLoader(dataset = X_train, batch_size = 10, shuffle = True)
###################################################################

learning_rate = 1e-4

in_size = height*width*3

reduction=1

#model = nn.Sequential(
#    nn.Linear(in_size,int(in_size/reduction)),
#    nn.ReLU(True),
#    # coded is here
#    nn.Linear(int(in_size/reduction), in_size),
#)

model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1, stride=1)
        )

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)
accuracies = []
epochs=3000
loss_fn = nn.MSELoss()

loss_list=[]

for e in range(epochs):

    for step, x in enumerate(train_loader):
        #model.train()  # put model to training mode
        
        #batch = np.random.choice(X.shape[0], 50)
        
        #batch_x  = x.view(-1, in_size)
        
        
        #reconstructed = model(X[batch])
        #reconstructed = model(batch_x)
        reconstructed = model(x.transpose(1,3))
        
        #loss = loss_fn(reconstructed, batch_x)
        loss = loss_fn(reconstructed, x.transpose(1,3))
        #print(loss)
        print('epoch: ', e, 'step: ', step, 'loss: ', loss.item())   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_list.append(loss.item())
        
        
        
   # print('epoch: ', e, 'step: ', step, 'loss: ', loss)        


print(X_train.shape)
img, orig_img = check_image(model, X_train, 3, height, width)
    
    
    