from load_images import load_images
import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
import torch
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from autoencoder_compress_gpu import linear_autoencoder_gpu
#from autoencoder_compress import linear_autoencoder
from pca_compress import PCA_Compress

plt.switch_backend('agg')

def msssim(img1, img2):
    dim = int(np.sqrt(img1.shape[0]/3))
    img1 = img1.reshape(dim, dim, 3).T
    img2 = img2.reshape(dim, dim, 3).T

    return pytorch_msssim.msssim(torch.from_numpy(img1[np.newaxis,:,:,:]).float(),
                          torch.from_numpy(img2[np.newaxis,:,:,:]).float()).item()

def mse(img1, img2):
    return mean_squared_error(img1, img2)

def msssim_batch(X, reconstructed):
    msssim_score = [msssim(X[i], reconstructed[i]) for i in range(X.shape[0])]
    msssim_score = np.mean([x for x in msssim_score if not math.isnan(x)])
    return msssim_score

def mse_batch(X, reconstructed):

    mse_score = np.mean([mse(X[i], reconstructed[i]) for i in range(X.shape[0])])
    return mse_score





def show(img_vec):
    print(img_vec.shape[0])
    print((np.sqrt(img_vec.shape[0]/3)))

    dim = int(np.sqrt(img_vec.shape[0]/3))
    print(dim)
    img = img_vec.reshape(dim, dim, 3)
    plt.imshow(img.astype(np.uint8))

def msssim(img1, img2):
    dim = int(np.sqrt(img1.shape[0]/3))
    img1 = img1.reshape(dim, dim, 3).T
    img2 = img2.reshape(dim, dim, 3).T

    return pytorch_msssim.msssim(torch.from_numpy(img1[np.newaxis,:,:,:]).float(),
                          torch.from_numpy(img2[np.newaxis,:,:,:]).float()).item()

def mse(img1, img2):
    return mean_squared_error(img1, img2)





################### LOAD DATA ###################

num_samples = 5000

dim = 100
#percent = 0.01
#X = load_images(num_samples, dim, dim)
X = load_images(num_samples, dim, dim).reshape(num_samples,-1)
X_val = X[4500:]
X=X[:4500]

print(X.shape)

##################### PCA #####################

run_pca=False

if(run_pca):
    percentages = [0.01, .02, .03, .04, .05, .06, .07, .08, .09]
    for p in percentages:

        total=int(dim*dim*3 * p)

        #X = load_images(num_samples, dim, dim).reshape(num_samples,-1)

        pca = PCA_Compress()



        compressed = pca.compress(X, total)

        print(compressed.shape)

        reconstructed = pca.uncompress(compressed)

        print(reconstructed.shape)

        show(reconstructed[0])

        print('percentage: ', p)
        #print(msssim(X[0], reconstructed[0]))




################ AUTOENCODER ###################


run_auto=True

if(run_auto):

    model = linear_autoencoder_gpu(X.shape[1], 0.01)

    model.cuda()
    model.train(X, epochs=1000)

    #print(type(model))
    #print(type(X))
    compressed = model.compress(Variable(torch.from_numpy(X_val)))
    #print(type(compressed))
    reconstructed = model.uncompress(compressed)
    #print(type(reconstructed))

    print('MSE')
    print('mse', mse_batch(X_val, reconstructed.data.cpu().numpy()))

    #show(reconstructed[0])

#    msssim_score = [msssim(X[i], reconstructed[i]) for i in range(num_samples)]
#    msssim_score = np.mean([x for x in msssim_score if x!=NaN])
#
#    print('MSSSIM Score:', msssim_score)
#    print('MSE Score:', mse(X[0], reconstructed[0]))
