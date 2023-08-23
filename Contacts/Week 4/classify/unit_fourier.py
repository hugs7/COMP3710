"""
Unit test for Fourier Transforms in pytorch
"""

import torch
import torchvision
import torchvision.transforms as transforms
import time

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Parameters
path = ".Contacts\\Week 4\\classify\\"

# ---------

# Data

# Convert each image into a tensor (from a np array)
# Normalise with mean = 0 and variance/standard deviation = 1
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0))]
)

trainset = torchvision.datasets.MNIST(
    root=path + "data/mnist", train=True, download=True, transform=transform
)

# Load in train set of data. Has images all with associated labels
# Specify data loader
# Buffer controls how many images we pass to the GPU
# Parameter depends on GPU hardware, model, and data
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Process data
print("Processing data")
start_time = time.time()
Fx = 0
Fy = 0
# Iterate over the trainer set
for i, (x, y) in enumerate(train_loader):
    # x is image, y is label
    print(x.shape)  # 4D tensor. (#batches, channel, height, width)
    print(x.dtype)
    Fx = x.numpy()
    Fy = y.numpy()
    break  # Not going to iterate for now. We're going to plot the first one
epoch_time = time.time() - start_time
print(f"Took {epoch_time} seconds or {epoch_time/60} minutes to process")


# Plot processed first trained image
