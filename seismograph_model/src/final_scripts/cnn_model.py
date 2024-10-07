import torch
import torch.nn as nn

class SpectrogramArrivalCNN(nn.Module):
    def __init__(self):
        super(SpectrogramArrivalCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer (initialized later based on input size)
        self.fc1 = None

        # Output layer for predicting arrival time
        self.fc_time = nn.Linear(128, 1)

    def forward(self, x):
        # First convolutional layer
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # Second convolutional layer
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Dynamically set fully connected layer based on input size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)

        # Pass through the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # Predict arrival time
        time_output = self.fc_time(x)

        return time_output