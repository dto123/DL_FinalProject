import random
import numpy as np
#from cs682.data_utils import load_CIFAR10


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import matplotlib.image as image
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
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

def unflatten(x):
    N = x.shape[0]
    return x.view(N, 3, 50, 50)

class Unflatten(nn.Module):

    def forward(self, x):
        return unflatten(x)

plt.switch_backend('agg')

dtype = torch.cuda.FloatTensor

height, width = 50, 50

X_train = torch.from_numpy(load_images(100,height,width)).float()
#X_train = X_train.cuda()
print(X_train.size())

print('Done loading')

train_loader = Data.DataLoader(dataset = X_train, batch_size = 64, shuffle = True)
###################################################################

learning_rate = 1e-4

in_size = height*width*3
reduction=2

model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1, stride=1),
        Flatten(),
        nn.Linear(in_size, int(in_size/2)),
        nn.Linear(int(in_size/2), in_size),
        Unflatten()
    )

model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.9)
accuracies = []
epochs=1500
loss_fn = nn.MSELoss()

print('Start training')

for e in range(epochs):

    for step, x in enumerate(train_loader):
        #model.train()  # put model to training mode

        #batch = np.random.choice(X.shape[0], 50)

        x = Variable(x)

        #batch_x  = x.view(-1, in_size)
        #print(type(batch_x))
        #batch_x = batch_x.cuda()
        #print(batch_x)
        #reconstructed = model(X[batch])
        reconstructed = model(x.transpose(1,3).cuda())
        #print(type(reconstructed))
        model = model.cuda()
        loss = loss_fn(reconstructed, x.transpose(1,3).cuda())
        #print(loss)
        print('epoch: ', e, 'step: ', step, 'loss: ', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#reconstructed = model(X_train.transpose(1,3)).transpose(1,3)
#reconstructed.clamp_(0,255)

final = reconstructed.data.cpu().clamp_(0,255).transpose(1,3).numpy()
print(type(final))
print(final.shape)

final_6 = final[6].reshape(height, width,3)
print(type(final_6))
print(final_6.shape)
plt.imshow(final_6)
plt.savefig('new_image_6_50by50.png')

orig = X_train[6]
plt.imshow(orig)
plt.savefig('orig_image_6_50by50.png')




   # print('epoch: ', e, 'step: ', step, 'loss: ', loss)
def check_image(model, X, ind, height, width):

    reconstructed = model(X)
    img = reconstructed.detach().numpy()[ind].reshape(height, width,3)
    orig_img = X.detach().numpy()[ind].reshape(height, width,3)
    plt.imshow(img)
    plt.savefig('image_4_10by10.png')
    #plt2.imshow(orig_img)

#check_image(model, X_train, 4, height, width)
