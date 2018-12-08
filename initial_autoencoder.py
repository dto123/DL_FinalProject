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

dtype = torch.cuda.FloatTensor

height, width = 10, 10

X_train = torch.from_numpy(load_images(100,height,width)).float()
#X_train = X_train.cuda()

print('Done loading')

train_loader = Data.DataLoader(dataset = X_train, batch_size = 64, shuffle = True)
###################################################################

learning_rate = 1e-4

in_size = height*width
reduction=1

model = nn.Sequential(
    nn.Linear(in_size,int(in_size/reduction)),
    #nn.ReLU(True),
    # coded is here
    nn.Linear(int(in_size/reduction), in_size)
)

model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.9)
accuracies = []
epochs=100
loss_fn = nn.MSELoss()

print('Start training')

for e in range(epochs):

    for step, x in enumerate(train_loader):
        #model.train()  # put model to training mode

        #batch = np.random.choice(X.shape[0], 50)

        x = Variable(x)

        batch_x  = x.view(-1, in_size)
        #print(type(batch_x))
        #batch_x = batch_x.cuda()
        #print(batch_x)
        #reconstructed = model(X[batch])
        reconstructed = model(batch_x.cuda())
        #print(type(reconstructed))
        model = model.cuda()
        loss = loss_fn(reconstructed, batch_x.cuda())
        #print(loss)
        print('epoch: ', e, 'step: ', step, 'loss: ', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

final = reconstructed.numpy()
print(type(final))
plt.imshow(final[4])
print(type(final[4]))
plt.savefig('image_4_10by10.png')





   # print('epoch: ', e, 'step: ', step, 'loss: ', loss)
def check_image(model, X, ind, height, width):

    reconstructed = model(X)
    img = reconstructed.detach().numpy()[ind].reshape(height, width,3)
    orig_img = X.detach().numpy()[ind].reshape(height, width,3)
    plt.imshow(img)
    plt.savefig('image_4_10by10.png')
    #plt2.imshow(orig_img)

#check_image(model, X_train, 4, height, width)
