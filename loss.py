import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

BCE_loss = nn.BCELoss()

def loss_function(x,# pred
                  x_hat, # target
                  mean, # mean encode
                  log_var): # variance encode
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        # get p(z|x) as z, z forward decoder to compute pred (as x_hat) as input such that as like as p(x|z) (as x)             
    KLD      = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # log_var, mean is encoder output

                    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return reproduction_loss + KLD


