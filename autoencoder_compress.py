#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:22:43 2018

@author: eric
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class linear_autoencoder(nn.Module):
    def __init__(self, size, reduction):
        super(linear_autoencoder, self).__init__()
        self.size=size
        self.reduction=reduction
        self.encoder = nn.Sequential(
            nn.Linear(size, int(size*reduction)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(size*reduction), size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compress(self, x):
        print(type(x))
        x = x.data
        return self.encoder(x.float()).detach().numpy()

    def uncompress(self, x):
        return self.decoder(torch.from_numpy(x).float()).detach().numpy()

    def train(self, X, epochs=100):

        X = torch.from_numpy(X).float()
        train_loader = Data.DataLoader(dataset = X, batch_size = 64, shuffle = True)

        learning_rate = 1e-4


        # define optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.9)
        loss_fn = nn.MSELoss()

        for e in range(epochs):

            for step, x in enumerate(train_loader):
                #model.train()  # put model to training mode

                reconstructed = self.forward(x)

                #loss = loss_fn(reconstructed, batch_x)
                loss = loss_fn(reconstructed, x)
                #print(loss)
                print('epoch: ', e, 'step: ', step, 'loss: ', loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
