from utils import linear_beta_schedule, cosine_beta_schedule
import torch
import torch.nn.functional as F
T = 200
betas = cosine_beta_schedule(timesteps=T)
betas = linear_beta_schedule(timesteps=T)
#PreCalculated
alphas = 1. -betas
alphas_cumprod = torch.cumprod(alphas, 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

IMG_SIZE = 64
BATCH_SIZE = 64