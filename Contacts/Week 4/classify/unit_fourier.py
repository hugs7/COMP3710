"""
Unit test for Fourier Transforms in pytorch
"""

from typing import Any
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Parameters
path = "\\Contacts\\Week 4\\classify\\"

# ---------


# Custom Data Transform
# In pytorch everything is a module :)
# To write a custom layer, model, transform, etc, we derive from nn.module class
class NormaliseStatistics(torch.nn.Module):
    """
    Normalise the data to sample zero mean, unit variance per channel
    """

    def __init__(self, unit_variance=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unit_variance = unit_variance

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        """
        Args:
                x (PIL Image or np.ndarray): Image to be transformed.
        Returns:
                Tensor: Transformed image.
        """
        means = torch.mean(x, dim=[-2, -1])
        if self.unit_variance:
            stddevs = torch.std(x, dim=[-2, -1])  # per channel
            x = (x - means[:, None, None]) / stddevs[:, None, None]
            # Only divide by standard deviation if call specified unit variance
        else:
            x = x - means[:, None, None]

        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


# Data

# Convert each image into a tensor (from a np array)
# Normalise with mean = 0 and variance/standard deviation = 1
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0))]
# )

# Now use custom Transform class instead
transform = transforms.Compose(
    [transforms.ToTensor(), NormaliseStatistics(unit_variance=True)]
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
plt.figure(figsize=(10, 10))

plot_size = 6  # 4x4 plot of all the data
for i in range(plot_size**2):
    # define subplot
    plt.subplot(plot_size, plot_size, 1 + i)
    # Turn axis off
    plt.axis("off")
    plt.tight_layout()

    # plot raw pixel data
    plt.imshow(np.abs(Fx[i, 0, :, :]))
    plt.title(str(Fy[i]))


plt.show()
