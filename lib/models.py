###########################################################################
## Adapted from Geometry regularized autoencoder (GRAE)
## https://github.com/KevinMoonLab/GRAE/blob/master/notebook/GRAE.ipynb
###########################################################################

import os
import shutil
import copy
from six.moves import cPickle as pickle #for performance

import warnings  # Ignore sklearn future warning
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import math
import scipy
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA, PCA
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

import time

from sklearn.model_selection import train_test_split
from skbio.stats.distance import mantel

# Datasets import
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as torch_datasets
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage
from scipy.stats import pearsonr
import scprep


import urllib
from scipy.io import loadmat

# Model imports
import umap
import phate

# Plots
import matplotlib.pyplot as plt

# Model parameters
BATCH = 200
LR = .0001
WEIGHT_DECAY = 1
EPOCHS = 200
SEED = 7512183

## AutoEncoder
# Vanilla AE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



# AE building blocks
class Encoder_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, z_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.mu = nn.Linear(hidden_dim3, z_dim)
    def forward(self, x):
        hidden1 = F.relu(self.linear(x))
        hidden2 = F.relu(self.linear2(hidden1))
        hidden3 = F.relu(self.linear3(hidden2))
        z_mu = self.mu(hidden3)
        return z_mu

class Decoder_MLP(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, sigmoid_act = False):
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.out = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid_act = sigmoid_act
    def forward(self, x):
        hidden1 = F.relu(self.linear(x))
        hidden2 = F.relu(self.linear2(hidden1))
        hidden3 = F.relu(self.linear3(hidden2))
        if self.sigmoid_act == False:
            predicted = (self.out(hidden3))
        else:
            predicted = F.sigmoid(self.out(hidden3))
        return predicted

class AE_MLP(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        latent = self.enc(x)
        predicted = self.dec(latent)
        return predicted, latent


# AE main class
class AE():
    """Autoencoder class with sklearn interface."""
    def __init__(self, input_size, random_state=SEED, track_rec=False,
                 AE_wrapper=AE_MLP, batch_size=BATCH, lr=LR,
                 weight_decay=WEIGHT_DECAY, reduction='sum', epochs=EPOCHS, **kwargs):
        layer_1 = 800
        layer_2 = 400
        layer_3 = 200
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.encoder = Encoder_MLP(input_size, layer_1, layer_2, layer_3, 2)
        self.decoder = Decoder_MLP(2, layer_3, layer_2, layer_1, input_size)
        self.model = AE_wrapper(self.encoder, self.decoder, **kwargs)
        self.model = self.model.float().to(device)

        self.criterion = nn.MSELoss(reduction=reduction)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr = self.lr,
                                        weight_decay=self.weight_decay)
        self.loss = list()
        self.track_rec = track_rec
        self.random_state = random_state

    def fit(self, x):
        # Train AE
        self.model.train()

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                           shuffle=True)

        for epoch in range(self.epochs):
            for batch in loader:
                data, y = batch
                data = data.to(device)

                self.optimizer.zero_grad()
                x_hat, _ = self.model(data)
                x_hat = x_hat.to(device)
                loss = self.criterion(data, x_hat)
                loss.backward()
                self.optimizer.step()

            if self.track_rec:
                x_np, _ = x.numpy()
                x_hat = self.inverse_transform(self.transform(x))
                self.loss.append(mean_squared_error(x_np, x_hat))

    def transform(self, x):
        self.model.eval()
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                           shuffle=False)
        z = [self.encoder(batch.to(device)).cpu().detach().numpy()
             for batch, _ in loader]

        return np.concatenate(z)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, z):
        self.model.eval()
        z = NumpyDataset(z)
        loader = torch.utils.data.DataLoader(z, batch_size=self.batch_size,
                                           shuffle=False)
        x_hat = [self.decoder(batch.to(device)).cpu().detach().numpy()
        for batch in loader]

        return np.concatenate(x_hat)


class ManifoldLoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.MSE = nn.MSELoss(reduction='sum')
        self.loss = None

    def forward(self, x, y, z, emb):
        self.loss = self.MSE(x, y) + self.lam * self.MSE(z, emb)
        return self.loss

    def backward(self):
        self.loss.backward()

    def decay_lam(self, factor):
        self.lam *= factor

# Datasets
# Base class
class BaseDataset(Dataset):
    """Simple class for X and Y ndarrays."""
    def __init__(self, x, y, split, split_ratio, seed):
        if split not in ('train', 'test', 'none'):
            raise Exception('split argument should be "train", "test" or "none"')

        x, y = self.get_split(x, y, split, split_ratio, seed)

        self.data = x.float()
        self.targets = y.float()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        if idx == None:
            return self.data.numpy(), self.targets.numpy()
        else:
            return self.data.numpy()[idx], self.targets.numpy()[idx]

    def get_split(self, x, y, split, split_ratio, seed):
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        n = x.shape[0]
        train_idx, test_idx = train_test_split(np.arange(n),
                                            train_size=split_ratio,
                                            random_state=seed)

        if split == 'train':
            return torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
        else:
            return torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx])



class NumpyDataset(Dataset):
    """Wrapper for x ndarray with no targets."""
    def __init__(self, x):
        self.data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        if idx == None:
            return self.data.numpy()
        else:
            return self.data.numpy()[idx]