import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

def flatten(x):
    N = x.shape[0]
    return x.transpose(1,3).contiguous().view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def unflatten(x):
    N = x.shape[0]
    square_dim = int(math.sqrt(x.shape[1]/3))
    return x.view(N, square_dim, square_dim, 3).transpose(1,3)

class Unflatten(nn.Module):
    def forward(self, x):
        return unflatten(x)


class linear_autoencoder_gpu(nn.Module):
    def __init__(self, size, reduction):
        super(linear_autoencoder_gpu, self).__init__()
        self.size=size
        self.reduction=reduction



        self.encoder = nn.Sequential(
            #nn.Linear(size, size),
            Unflatten(),
            nn.Conv2d(3, 3, kernel_size=2, stride=2),
            Flatten(),
            #nn.BatchNorm1d(size),
            #nn.ReLU(),
            nn.Linear(1875, int(size*reduction)),
            #nn.ReLU(),
            #nn.Linear(int(size*.5), int(size*.25)),
            #nn.ReLU(),
            #nn.Linear(int(size*.25), int(size*.125)),
            #nn.ReLU(),
            #nn.Linear(int(size*10*reduction), int(size*reduction)),
            #nn.Tanh(),
            #nn.Linear(int(size*reduction),int(size*reduction))

        )
        self.decoder = nn.Sequential(
            #nn.Linear(int(size*reduction), int(size*10*reduction)),
            #nn.Linear(int(size*10*reduction), size),
            nn.Linear(int(size*reduction), 1875),

            #nn.Linear(int(size*reduction), size),
            #nn.BatchNorm1d(int(size)),
            #nn.ReLU(),
            Unflatten(),
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2),
            Flatten(),
            #nn.ReLU(),
            #nn.Linear(size, size),
        )




    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compress(self, x):
        print('compress')
        print(type(x))
        return self.encoder(x.cuda().float())
    def uncompress(self, x): return self.decoder(x)

    def train(self, X, epochs=100):

        X = torch.from_numpy(X).float()
        train_loader = Data.DataLoader(dataset = X, batch_size = 64, shuffle = True)

        learning_rate = 1e-2


        # define optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0)
        loss_fn = nn.MSELoss()

        for e in range(epochs):

            for step, x in enumerate(train_loader):
                #model.train()  # put model to training mode
                x = Variable(x)

                reconstructed = self.forward(x.cuda())

                #loss = loss_fn(reconstructed, batch_x)
                loss = loss_fn(reconstructed, x.cuda())
                #print(loss)
                print('epoch: ', e, 'step: ', step, 'loss: ', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
