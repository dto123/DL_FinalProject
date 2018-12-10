
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

class linear_autoencoder_gpu(nn.Module):
    def __init__(self, size, reduction):
        super(linear_autoencoder_gpu, self).__init__()
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
        print('compress')
        print(type(x))
        return self.encoder(x)
    def uncompress(self, x): return self.decoder(x)

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
                x = Variable(x)

                reconstructed = self.forward(x.cuda())

                #loss = loss_fn(reconstructed, batch_x)
                loss = loss_fn(reconstructed, x.cuda())
                #print(loss)
                print('epoch: ', e, 'step: ', step, 'loss: ', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
