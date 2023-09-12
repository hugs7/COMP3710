import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Download the faces (70 per person for good training)
lfw_people = fetch_lfw_people(
    min_faces_per_person=100, resize=0.4, download_if_missing=True
)

# Extract parameters from faces
X = lfw_people.data
y = lfw_people.target
n_classes = len(np.unique(y))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)


# Define a CNN-based model
class CNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        # Pooling between fc layers to keep dimensions down
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 11 * 7, 128)  # Adjust the input size here
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(-1, 1, 50, 37)  # Reshape input for CNN
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model
model = CNNClassifier(n_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create a DataLoader for training and testing data
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        output = model(batch_x)
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.cpu().numpy())

# Calculate accuracy and print classification report
correct = np.sum(np.array(predictions) == y_test)
total_test = len(y_test)
accuracy = correct / total_test
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions, target_names=lfw_people.target_names))
