import config
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import linear_beta_schedule, cosine_beta_schedule, get_index_from_list, forward_diffusion_sample
import UNet as U_Net

class Defusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], attention=True, time_emb_dim=50):
        super(Defusion, self).__init__()
        self.UNet = U_Net.UNet_With_T(in_channels, out_channels, features, attention)
        self.time_emb = U_Net.SinusoidalPositionalEmbedding(time_emb_dim)

    def forward(self, x, t):
        return self.UNet(x,t)