"""
Demo 2
Part 3 - GAN
Hugo Burton - s4698512
11/09/2023
"""

from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch import flatten
from torch import nn

# import the necessary packages
from pyimagesearch.dcgan import Generator
from pyimagesearch.dcgan import Discriminator
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import numpy as np
import argparse
import torch
import cv2
import os

import time
import matplotlib.pyplot as plt


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


# Hyper-parameters
num_epochs = 2
learning_rate = 5e-3
channels = 3

# paths
path = "C:\\Users\\Hugo Burton\\OneDrive\\Documents\\University (2021 - 2024)\\2023 Semester 2\\COMP3710 Data\\"


# Generator class
class Generator(nn.Module):
    def __init__(self, inputDim=100, outputChannels=1):
        super(Generator, self).__init__()

        # first set of CONVT => RELU => BN
        self.ct1 = ConvTranspose2d(
            in_channels=inputDim,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False,
        )
        self.relu1 = ReLU()
        self.batchNorm1 = BatchNorm2d(128)

        # second set of CONVT => RELU => BN
        self.ct2 = ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu2 = ReLU()
        self.batchNorm2 = BatchNorm2d(64)

        # last set of CONVT => RELU => BN
        self.ct3 = ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu3 = ReLU()
        self.batchNorm3 = BatchNorm2d(32)

        # apply another upsample and transposed convolution, but
        # this time output the TANH activation
        self.ct4 = ConvTranspose2d(
            in_channels=32,
            out_channels=outputChannels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        self.tanh = Tanh()

    def forward(self, x):
        # pass the input through our first set of CONVT => RELU => BN
        # layers
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batchNorm1(x)

        # pass the output from previous layer through our second
        # CONVT => RELU => BN layer set
        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)

        # pass the output from previous layer through our last set
        # of CONVT => RELU => BN layers
        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchNorm3(x)

        # pass the output from previous layer through CONVT2D => TANH
        # layers to get our output
        x = self.ct4(x)
        output = self.tanh(x)

        # return the output
        return output


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super(Discriminator, self).__init__()

        # first set of CONV => RELU layers
        self.conv1 = Conv2d(
            in_channels=depth, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.leakyRelu1 = LeakyReLU(alpha, inplace=True)

        # second set of CONV => RELU layers
        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.leakyRelu2 = LeakyReLU(alpha, inplace=True)

        # first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=3136, out_features=512)
        self.leakyRelu3 = LeakyReLU(alpha, inplace=True)

        # sigmoid layer outputting a single value
        self.fc2 = Linear(in_features=512, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # pass the input through first set of CONV => RELU layers
        x = self.conv1(x)
        x = self.leakyRelu1(x)

        # pass the output from the previous layer through our second
        # set of CONV => RELU layers
        x = self.conv2(x)
        x = self.leakyRelu2(x)

        # flatten the output from the previous layer and pass it
        # through our first (and only) set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu3(x)

        # pass the output from the previous layer through our sigmoid
        # layer outputting a single value
        x = self.fc2(x)
        output = self.sigmoid(x)

        # return the output
        return output


# Training
# USAGE
# python dcgan_mnist.py --output output


# custom weights initialization called on generator and discriminator
def weights_init(model):
    # get the class name
    classname = model.__class__.__name__
    # check if the classname contains the word "conv"
    if classname.find("Conv") != -1:
        # intialize the weights from normal distribution
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    # otherwise, check if the name contains the word "BatcnNorm"
    elif classname.find("BatchNorm") != -1:
        # intialize the weights from normal distribution and set the
        # bias to 0
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# epochs to train for")
ap.add_argument(
    "-b", "--batch-size", type=int, default=128, help="batch size for training"
)
args = vars(ap.parse_args())
# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]


# set the device we will be using
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define data transforms
dataTransforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)
# load the MNIST dataset and stack the training and testing data
# points so we have additional training data
print("[INFO] loading MNIST dataset...")
trainData = MNIST(root="data", train=True, download=True, transform=dataTransforms)
testData = MNIST(root="data", train=False, download=True, transform=dataTransforms)
data = torch.utils.data.ConcatDataset((trainData, testData))
# initialize our dataloader
dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)


# calculate steps per epoch
stepsPerEpoch = len(dataloader.dataset) // BATCH_SIZE
# build the generator, initialize it's weights, and flash it to the
# current device
print("[INFO] building generator...")
gen = Generator(inputDim=100, outputChannels=1)
gen.apply(weights_init)
gen.to(DEVICE)
# build the discriminator, initialize it's weights, and flash it to
# the current device
print("[INFO] building discriminator...")
disc = Discriminator(depth=1)
disc.apply(weights_init)
disc.to(DEVICE)
# initialize optimizer for both generator and discriminator
genOpt = Adam(
    gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / NUM_EPOCHS
)
discOpt = Adam(
    disc.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / NUM_EPOCHS
)
# initialize BCELoss function
criterion = BCELoss()
