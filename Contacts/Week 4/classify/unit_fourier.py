"""
Unit test for Fourier Transforms in pytorch
"""

from typing import Any
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
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


# Fourier Transform


class Fourier(torch.nn.Module):
    """
    Discrete Fourier Transform of image and center
    Filter applies a filter to the coefficients, assumes center matches shape of filter
    Zero mean sets DC offset coefficient to epsilon > 0
    as_real_channels returns real and imaginary components as real channels
    """

    def __init__(self, centre, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.centre = centre
        self.zero_mean = True

    def __call__(self, x):
        """
        Args:
                x (PIL Image or np.ndarray): Image to be transformed.
        Returns:
                Tensor: Fourier Transformed image.
        """

        # Channel, height, width
        c, h, w = F.get_dimensions(x)

        # Make complex signal
        x_complex = x + 0j

        # Apply fft only to image dimensions (height and width)
        # If dim isn't specified, fft will be applied to all dimensions
        # and you'll end up with something weird
        Fx = torch.fft.fft2(x_complex, dim=(-2, -1))

        # When doing fast fourier transform, the zeroth element, the centre of the fourier transform if actually
        # the DC coefficient. DFT matrix has ones. Ones is basically a sum. Setting to a small epsilon
        # is equivalent to setting it to 0
        if self.zero_mean:
            # Remove DC coefficient
            Fx[..., 0, 0] = 1e-12

        if self.centre:
            # Compute fourier transform
            Fx = torch.fft.fftshift(Fx, dim=(-2, -1))

        return Fx


# Data

# torchvision transform
# Convert each image into a tensor (from a np array)
# Normalise with mean = 0 and variance/standard deviation = 1
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0))]
)

# Now use custom Transform class instead
custom_transform = transforms.Compose(
    [transforms.ToTensor(), NormaliseStatistics(unit_variance=True)]
)

fourier_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        NormaliseStatistics(unit_variance=True),
        Fourier(centre=True),
    ]
)

trainset = torchvision.datasets.MNIST(
    root=path + "data/mnist", train=True, download=True, transform=fourier_transform
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
print(
    "Took "
    + str(epoch_time)
    + " seconds or "
    + str(epoch_time / 60)
    + " minutes to process"
)


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
