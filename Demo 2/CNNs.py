"""
Demo 2
Part 2 - CNNs (6 marks)
Hugo Burton - s4698512
23/08/2023
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Download the faces (70 per person for good training)
lfw_people = fetch_lfw_people(
    min_faces_per_person=100, resize=0.4, download_if_missing=True
)

# Extract parameters from faces
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_featrues = X.shape[1]

# Labels
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Dataset size")
print(f"samples: n = {n_samples}")
print(f"features: n = {n_featrues}")
print(f"classes: n = {n_classes}")

# Split data into training and testing
# Testing set also should have a validation set
# This is so we can test / validate the model's performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Dimensionality reduction
n_components = 150

# Normalise data
X_train /= 255.0
X_test /= 255.0
X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]
print("X_train shape", X_train.shape)

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
