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
import numpy as np

"""
Create model
"""
NUM_CLASSES = 1000
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
#     num_classes = NUM_CLASSES,           # output number of classes
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
    num_classes = NUM_CLASSES,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)



weight_path = "/storage/home/hcocice1/nwitten3/weights_0260_cifar100.pth"
if os.path.exists(weight_path):
    print("Loading Weights")
    load_pretrained_weights(model, weight_path, "student", None, None)

    
    
"""
Create Data
"""

training_data = torchvision.datasets.CIFAR100(
    root = "data", 
    train = True, 
    transform = ToTensor(), 
    download = False)

test_data = torchvision.datasets.CIFAR100(
    root = "data", 
    train = False, 
    transform = ToTensor(), 
    target_transform = None, 
    download = False)

test_size = 100

# Instantiate data loaders
train_dataloader = DataLoader(training_data, batch_size=2000, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=test_size, shuffle=False)

# Load the first batch for both test and train
test_features, test_labels = next(iter(test_dataloader))
train_features, train_labels = next(iter(train_dataloader))

# Send to GPU
model = model.to(torch.device("cuda"))
test_features = test_features.to(torch.device("cuda"))
train_features = train_features.to(torch.device("cuda"))

# Run training data to get labeled latent space
train_output = model(train_features.permute((0,2,3,1)))
train_output = train_output.cpu().detach().numpy()
train_labels = train_labels.numpy()

# Now run test data through the model
test_output = model(test_features.permute((0,2,3,1)))
test_output = test_output.cpu().detach().numpy()
test_labels = test_labels.numpy()

# Now classifiy each test point by finding the closest training points
for k in [1, 5, 10, 50, 75, 100, 500, 1000]:
    correct = 0
    for output_vector, true_label in zip(test_output, test_labels):
        diff_vectors = train_output - output_vector
        distances = np.linalg.norm(diff_vectors, axis=1)
        # Find the closest K points
        closest_idxs = np.argsort(distances)[:k]
        closest_labels = train_labels[closest_idxs]
        # Count the most frequent label in these K points
        values, counts = np.unique(closest_labels, return_counts=True)
        label = values[np.argmax(counts)]
        if label == true_label:
            correct += 1

    print(f"{k}-NN Accuracy: {correct / test_size * 100}%")
