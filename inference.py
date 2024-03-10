import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from model import Encoder, Decoder, Model
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from loss import loss_function
import matplotlib.pyplot as plt


cuda = True
DEVICE = torch.device("cuda" if cuda == True else "cpu")
batch_size = 100
latent_dim = 200
x_dim  = 784
hidden_dim = 400

decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)


with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)


save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())