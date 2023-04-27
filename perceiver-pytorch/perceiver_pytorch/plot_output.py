# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:18:20 2023

@author: jvbre
"""

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from perceiver_pytorch import Perceiver
import matplotlib.pyplot as plt
import os
from utils import load_pretrained_weights
from sklearn.decomposition import PCA

"""
Create model
"""
OUTPUT_SIZE = 1000
NUM_CLASSES = 10
# model = Perceiver(
#     input_channels = 3,          # number of channels for each token of the input
#     input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
#     num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
#     max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
#     depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
#                                  #   depth * (cross attention -> self_per_cross_attn * self attention)
#     num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
#     latent_dim = 512,            # latent dimension
#     cross_heads = 1,             # number of heads for cross attention. paper said 1
#     latent_heads = 8,            # number of heads for latent self attention, 8
#     cross_dim_head = 64,         # number of dimensions per cross attention head
#     latent_dim_head = 64,        # number of dimensions per latent self attention head
#     num_classes = OUTPUT_SIZE,           # output number of classes
#     attn_dropout = 0.,
#     ff_dropout = 0.,
#     weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
#     fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
#     self_per_cross_attn = 2      # number of self attention blocks per cross attention
# )
model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 2,                   # depth of net. The shape of the final attention mechanism will be:
                                #   depth * (cross attention -> self_per_cross_attn * self attention)
    num_latents = 64,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 64,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    num_classes = OUTPUT_SIZE,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)



weight_path = "/storage/home/hcocice1/nwitten3/weights.pth"
if os.path.exists(weight_path):
    print("Loading Weights")
    load_pretrained_weights(model, weight_path, "student", None, None)

    
    
"""
Create Data
"""

# training_data = torchvision.datasets.CIFAR10(
#     root = "data", 
#     train = True, 
#     transform = ToTensor(), 
#     download = True)

test_data = torchvision.datasets.CIFAR10(
    root = "data", 
    train = False, 
    transform = ToTensor(), 
    target_transform = None, 
    download = False)


# train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)

test_features, test_labels = next(iter(test_dataloader))

model = model.to(torch.device("cuda"))
test_features = test_features.to(torch.device("cuda"))
output = model(test_features.permute((0,2,3,1)))

output = output.cpu()
output = output.detach().numpy()

pca = PCA(n_components=2)
pca.fit(output)
output = pca.transform(output)

test_labels = test_labels.numpy()
print(output)
print(output.shape)
plt.figure(figsize=(10, 10))
plt.xlim([-5, 5])
plt.ylim([-5, 5])
# for i in range(NUM_CLASSES):
for i in [6, 8]:
    mask = (test_labels == i)
    outputs_for_class = output[mask]
    x = []
    y = []
    if len(outputs_for_class):
        x = outputs_for_class[:, 0]
        y = outputs_for_class[:, 1]
    # plt.subplot(5,2,i+1)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    plt.scatter(x, y)
plt.tight_layout()
plt.savefig('/storage/home/hcocice1/nwitten3/output.png')


