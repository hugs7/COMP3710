"""
Demo 2
Part 2 - CNNs (6 marks)
Hugo Burton - s4698512
23/08/2023
"""

# Machine Learning
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import VGG

import BasicBlock

from torch.cuda.amp import autocast


# Plotting
import matplotlib.pyplot as plt
import numpy as np

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
num_epochs = 35
learning_rate = 0.1
num_classes = 10
channels = 3

# paths
path = "C:\\Users\\Hugo Burton\\OneDrive\\Documents\\University (2021 - 2024)\\2023 Semester 2\\COMP3710 Data\\"

# --------------
# Data

# Define transforms for the training set and testing set separately
# Add "additional data" to the training set by random transformations
# both crop and horizontal flip.
transform_training = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    ]
)

transform_testing = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)

# Training data
trainset = torchvision.datasets.CIFAR10(
    root=path + "data/cifar10", train=True, download=False, transform=transform_training
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True
)  # num_workers=6

total_step = len(train_loader)

# Testing data
testset = torchvision.datasets.CIFAR10(
    root=path + "data/cifar10", train=False, download=False, transform=transform_testing
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False
)  # num_workers=6


# Model

model = BasicBlock.ResNet18()
model = model.to(device)

# model info
print("Model parameters:", sum([p.nelement() for p in model.parameters()]))
# print(model)

# Loss function (criterion) measures how close the model is to the true value (during training)
criterion = nn.CrossEntropyLoss()

# Optimise the model with a learning rate alpha as 5e-3. Like a time step to an optimisation problem
# SGD is stochastic gradient descent
optimiser = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
)

# Learning rate schedule (changes over time)
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser,
    max_lr=learning_rate,
    total_steps=total_step,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
)

# Train Model

# Place module in training mode
model.train()
print("> Training")
start_train = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimiser.zero_grad()

        with autocast():
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

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

    scheduler.step()

end_train = time.time()
elapsed_training = end_train - start_train

print("Training took " + str(elapsed_training) + " seconds.")


# Test Model
print("> Testing")
start_test = time.time()
# Place module in evaluation mode
model.eval()
true_labels_all = []
predicted_all = []
images_all = []
with torch.no_grad():
    with autocast():
        correct = 0
        total = 0
        # Iterate over the images in each batch provided by the test loader
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Store images in an array so I can access them later
            images_all.append(images)
            true_labels_all.append(labels)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            # Store predicted labels in an array so I can access them later
            predicted_all.append(predicted)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test Accuracy: {} %".format(100 * correct / total))

end_test = time.time()
elapsed_testing = end_test - start_test
print("Testing took " + str(elapsed_testing) + " seconds.")


# Plot results
# Choose random testing set to print
import random

rand = random.randint(0, 39)  # Can't include 40 as it's shorter
print("Predicted Labels: ", predicted_all[rand])
# Convert labels into a number
predicted_labels = [p.item() for p in predicted_all[rand]]
# Get images from same
plot_images = images_all[rand]
# Get true labels
true_labels = [p.item() for p in true_labels_all[rand]]

# Define class labels for CIFAR-10 dataset
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Create a 15x8 grid of subplots
fig, axes = plt.subplots(6, 20, figsize=(25, 25))

# Flatten the axes array for easy indexing
axes = axes.ravel()

# Assuming img is your normalized image
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


# Check this normalisation works.

for i in range(20 * 6):
    ax = axes[i]
    ax.axis("off")  # Turn off axis
    if predicted_labels[i] == true_labels[i]:
        colour = "black"
    else:
        colour = "red"
    ax.set_title(
        classes[predicted_labels[i]] + " (" + classes[true_labels[i]] + ")",
        color=colour,
    )  # Set title color to red

    # Apply the reverse normalization here
    img = np.transpose(plot_images[i].cpu().numpy(), (1, 2, 0))
    for j in range(3):  # Iterate over the channels (R, G, B)
        img[:, :, j] = img[:, :, j] * std[j] + mean[j]
    img = np.clip(img, 0, 1)
    ax.imshow(img)


# # Loop through the images and display them in the subplots
# for i in range(20 * 6):
#     ax = axes[i]
#     ax.axis("off")  # Turn off axis
#     if predicted_labels[i] == true_labels[i]:
#         colour = "black"
#     else:
#         colour = "red"
#     ax.set_title(
#         classes[predicted_labels[i]] + " (" + classes[true_labels[i]] + ")",
#         color=colour,
#     )  # Set title color to red
#     ax.imshow(
#         np.transpose(plot_images[i].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5
#     )  # Unnormalize and display image

plt.tight_layout()
plt.show()
