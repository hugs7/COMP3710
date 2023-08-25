"""
Demo 2
Part 2 - CNNs (6 marks)
Hugo Burton - s4698512
23/08/2023
"""

from sklearn.datasets import fetch_lfw_people
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import VGG

"""
Construct a Fast CIFAR10 dataset classification network using one of the TF
/Keras/PyTorch/JAX to meet the standards of the DAWNBench Challenge as the 
following:
1.  The model must achieve more than 93% accuracy and that is trainable in the
    fastest time possible (usually under 30 mins on a cluster).
2.  Usage of pre-built models will generally not be allowed unless approved by 
    the demonstrator.
3.  The model must run inference or a single epoch of training on the Ranpur 
    compute cluster (see appendix A for details) during the demonstration.


Using a ResNet-18 and mixed precision, it is possible to achieve 94% on a model trained for only ap-
proximately 360 seconds on a NVIDIA V100 GPU on the cluster. Can you achieve a time that is equiv-
alent or faster? See appendix B for more resources on the DAWNBench challenge. (5 Marks)
5.
"""

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


# Hyper-parameters
num_epochs = 5
learning_rate = 5e-3
channels = 3

# paths
path = "C:\\Users\\Hugo Burton\\OneDrive\\Documents\\University (2021 - 2024)\\2023 Semester 2\\COMP3710 Data\\"


# --------------
# Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
)

# Training data
trainset = torchvision.datasets.CIFAR10(
    root=path + "data/cifar10", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=512, shuffle=True
)  # num_workers=6

total_step = len(train_loader)

# Testing data
testset = torchvision.datasets.CIFAR10(
    root=path + "data/cifar10", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False
)  # num_workers=6


# Model

model = VGG.VGG("VGG11_2", channels, num_classes=10)
model = model.to(device)

# model info
print("Model parameters:", sum([p.nelement() for p in model.parameters()]))
print(model)

# Loss function (criterion) measures how close the model is to the true value (during training)
criteroin = nn.CrossEntropyLoss()

# Optimise the model with a learning rate alpha as 5e-3. Like a time step to an optimisation problem
# SGD is stochastic gradient descent
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

# Train Model

model.train()
print("> Training")
start_train = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criteroin(outputs, labels)

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Print info about ever 100 epochs
        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

end_train = time.time()
elapsed = end_train - start_train

print("Trainig took " + str(elapsed) + " seconds.")
