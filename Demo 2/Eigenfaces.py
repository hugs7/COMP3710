"""
Demo 2
Part 1 - Eigenfaces (4 marks)
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

# Normalise data by subtracting the mean
mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

# Compute eigen-decomposition of X after normalising the data (subtract mean)
U, S, V = np.linalg.svd(X_train, full_matrices=False)
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# Project into PCA subspace
X_transformed = np.dot(X_train, components.T)
print(X_transformed.shape)
X_test_transformed = np.dot(X_test, components.T)
print(X_test_transformed.shape)


# Plot the resulting eigen-vectors of the face PCA model


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))

    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)

    # For the number of plots
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)))  # Can specify cmap here if I want
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


eigenface_titles = [f"eigenfaces {d}" for d in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# Evaluate performance of dimensionalitty reduction via compactness

# Variance that can be explained with the data we have. Proportional to 1 / samples
explained_variance = (S**2) / (n_samples - 1)

# Total variance
total_var = explained_variance.sum()

# Ratio of explained to total variance
explained_variance_ratio = explained_variance / total_var

ratio_cumsum = np.cumsum(explained_variance_ratio)
print(ratio_cumsum.shape)
eigenValueCount = np.arange(n_components)

plt.plot(eigenValueCount, ratio_cumsum[:n_components])
plt.title("Compactness")
plt.show()

# Build random forest classifier to classify the faces according to
# the labels

estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train)
predictions = estimator.predict(X_test_transformed)
correct = predictions == y_test
total_test = len(X_test_transformed)

print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct", correct)
print("Total Correct", np.sum(correct))
print("Accuracy", np.sum(correct) / total_test)

print(classification_report(y_test, predictions, target_names=target_names))
