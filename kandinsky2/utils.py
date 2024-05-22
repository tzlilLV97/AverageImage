import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import importlib
from .model.utils import get_named_beta_schedule, _extract_into_tensor
from copy import deepcopy
from sklearn.decomposition import PCA

def rand_prob(length, model_dtype=torch.float32, device="cpu"):
    #Generate a random probability distrubution of the length length , the sum of the elements is 1
    prob = np.random.rand(length)
    prob = prob / np.sum(prob)
    return torch.tensor(prob).to(device, dtype=model_dtype)

def custom_PCA(x, PCA_components,  weights,device="cpu", model_dtype=torch.float32, which_to_keep=[], which_to_reconstruct=[], amplify=1.0, which_to_amplify=[]):
    #print all the details about the operation without weights and x and PCA_components and model_dtype and device
    print(f"Performing PCA with {PCA_components} components, which to keep is {which_to_keep}, which to reconstruct is {which_to_reconstruct}, amplify is {amplify}, which to amplify is {which_to_amplify}")
    old_x = x
    sanity_chk = torch.sum(x * torch.tensor(weights).unsqueeze(1).to(device, dtype=torch.float32), dim=0)
    if which_to_keep != [] and max(which_to_keep) >= PCA_components:
        raise ValueError("The maximum of which_to_keep should be less than PCA_components")
    x = x.to(device, dtype=torch.float32)
    pca = PCA(n_components=PCA_components)
    image_embeddings = x.cpu().numpy()
    image_embeddings = pca.fit_transform(image_embeddings)
    if which_to_keep == []:
        which_to_keep = list(range(PCA_components))
    if which_to_amplify != []:
        for i in which_to_amplify:
            print(f"Amplifying the value {image_embeddings[:, i]} by {amplify}")
            image_embeddings[:,i] = amplify * image_embeddings[:,i]
    image_embeddings = image_embeddings[:, which_to_keep]
    reconstructed_embeddings = image_embeddings @ pca.components_[which_to_keep, :] + pca.mean_
    image_emb = torch.tensor(reconstructed_embeddings).to(device, dtype=torch.float32)
    if which_to_reconstruct != []:
        image_emb = image_emb[which_to_reconstruct,:].to(device, dtype=model_dtype)
        return image_emb
        image_emb = image_emb[:, which_to_reconstruct].to(device, dtype=model_dtype)
    image_emb = torch.mean(image_emb, dim=0)
    image_emb = image_emb.to(device, dtype=model_dtype)
    return image_emb
def custom_PCA_2(x, PCA_components,  weights,device="cpu", model_dtype=torch.float32, which_to_keep=[], which_to_reconstruct=[], amplify=1.0, which_to_amplify=[]):
    print(f"Performing PCA with {PCA_components} components, which to keep is {which_to_keep}, which to reconstruct is {which_to_reconstruct}, amplify is {amplify}, which to amplify is {which_to_amplify}")
    if amplify == 1.0:
        return custom_PCA(x, PCA_components,  weights,device, model_dtype, which_to_keep, which_to_reconstruct=which_to_amplify)
    old_x = x
    sanity_chk = torch.sum(x * torch.tensor(weights).unsqueeze(1).to(device, dtype=torch.float32), dim=0)
    if which_to_keep != [] and max(which_to_keep) >= PCA_components:
        raise ValueError("The maximum of which_to_keep should be less than PCA_components")
    x = x.to(device, dtype=torch.float32)
    pca = PCA(n_components=PCA_components)
    image_embeddings = x.cpu().numpy()
#    PCA.fit(image_embeddings.T)
    image_embeddings = pca.fit_transform(image_embeddings)
    if which_to_keep == []:
    #    print(f"Which to keep is {which_to_keep}", end=",")
        which_to_keep = list(range(PCA_components))
    image_embeddings = image_embeddings[:, which_to_keep]
    #Sanity check
    before = torch.tensor(image_embeddings).to(device, dtype=model_dtype)
    #print the coordinates to amplify

    if which_to_amplify != []:
        for i in which_to_amplify:
      #      print(f"Amplifying the value {image_embeddings[:, i]} by {amplify}")
            image_embeddings[:,i] = amplify * image_embeddings[:,i]
       #     print(f"Amplified the value {image_embeddings[:,i]} by {amplify}")
   #   #      print(f"Amplified the component {i} by {amplify}, ")
    after = torch.tensor(image_embeddings).to(device, dtype=model_dtype)
    #Print MSE between before and after
   # print(f"MSE between the before and after amplification: {torch.nn.functional.mse_loss(before, after)}")
    reconstructed_embeddings = image_embeddings @ pca.components_[which_to_keep, :] + pca.mean_
    image_emb = torch.tensor(reconstructed_embeddings).to(device, dtype=torch.float32)
    #TODO doing average now, change to weights later.
    image_emb = torch.mean(image_emb, dim=0)
    image_emb = image_emb.to(device, dtype=model_dtype)
    return image_emb


# def custom_PCA(x, PCA_components, weights, device="cpu", model_dtype=torch.float32, which_to_keep=[]):
#     old_x = x
#     sanity_chk = torch.sum(x * torch.tensor(weights).unsqueeze(1).to(device, dtype=torch.float32), dim=0)
#     if which_to_keep != [] and max(which_to_keep) >= PCA_components:
#         raise ValueError("The maximum of which_to_keep should be less than PCA_components")
#     x = x.to(device, dtype=torch.float32)
#
#     # Convert torch tensor to numpy array
#     image_embeddings = x.cpu().numpy()
#
#     # Perform PCA compression
#     pca = PCA(n_components=PCA_components)
#     pca.fit(image_embeddings)
#     print("Explained variance ratio: ", pca.explained_variance_ratio_)
#     print("Explained variance: ", pca.explained_variance_)
#     print("Singular values: ", pca.singular_values_)
#     print("Mean: ", pca.mean_)
#     print("Components: ", pca.components_)
#     print("Noise variance: ", pca.noise_variance_)
#     compressed_embeddings = pca.transform(image_embeddings)[:, which_to_keep]
#
#     # Inverse transform to get the reconstructed data
#     reconstructed_embeddings = pca.inverse_transform(compressed_embeddings)
#
#     # Convert reconstructed embeddings back to torch tensor
#     reconstructed_x = torch.tensor(reconstructed_embeddings, device=device, dtype=model_dtype)
#
#     # Perform the weighted sum
#     image_emb = torch.sum(reconstructed_x * torch.tensor(weights).unsqueeze(1).to(device, dtype=model_dtype), dim=0)
#     print("MSE between the original and the reconstructed from PCA: ", nn.functional.mse_loss(sanity_chk, image_emb))
#
#     return image_emb

def Lower_res_PCA(x, PCA_components,  weights,device="cpu", model_dtype=torch.float32, which_to_keep=[]):
    if which_to_keep != [] and max(which_to_keep) >= PCA_components:
        raise ValueError("The maximum of which_to_keep should be less than PCA_components")
    pca = PCA(n_components=PCA_components)
    image_embeddings = x.cpu().numpy()
    image_embeddings = pca.fit_transform(image_embeddings.T)
    if which_to_keep == []:
        which_to_keep = list(range(PCA_components))
    image_embeddings = image_embeddings[:, which_to_keep]
    reconstructed_embeddings = image_embeddings @ pca.components_[which_to_keep, :] + pca.mean_
    image_emb = torch.tensor(reconstructed_embeddings).to(device, dtype=model_dtype).T
    #Perform the weighted sum
    image_emb = torch.sum(image_emb * torch.tensor(weights).unsqueeze(1).to(device, dtype=model_dtype), dim=0)
    return image_emb
def prepare_mask(mask):
    mask = mask.float()[0]
    old_mask = deepcopy(mask)
    for i in range(mask.shape[1]):
        for j in range(mask.shape[2]):
            if old_mask[0][i][j] == 1:
                continue
            if i != 0:
                mask[:, i - 1, j] = 0
            if j != 0:
                mask[:, i, j - 1] = 0
            if i != 0 and j != 0:
                mask[:, i - 1, j - 1] = 0
            if i != mask.shape[1] - 1:
                mask[:, i + 1, j] = 0
            if j != mask.shape[2] - 1:
                mask[:, i, j + 1] = 0
            if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                mask[:, i + 1, j + 1] = 0
    return mask.unsqueeze(0)


def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


def q_sample(x_start, t, schedule_name="linear", num_steps=1000, noise=None):
    betas = get_named_beta_schedule(schedule_name, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    if noise is None:
        noise = torch.randn_like(x_start)
    assert noise.shape == x_start.shape
    return (
        _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


def process_images(batch):
    scaled = (
        ((batch + 1) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .to("cpu")
        .permute(0, 2, 3, 1)
        .numpy()
    )
    images = []
    for i in range(scaled.shape[0]):
        images.append(Image.fromarray(scaled[i]))
    return images
