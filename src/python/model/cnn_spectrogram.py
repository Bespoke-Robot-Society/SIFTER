import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json


def self_train_on_martian_data(
    model,
    martian_data_loader,
    optimizer,
    criterion_event,
    criterion_time,
    num_epochs=10,
):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in martian_data_loader:
            images = batch[0]  # Unpack the images from the batch tuple
            optimizer.zero_grad()

            # Forward pass through the model
            event_output, time_output = model(images)

            # Generate pseudo-labels (predicted event labels)
            _, pseudo_labels = torch.max(event_output, 1)

            # Compute loss using pseudo-labels
            loss_event = criterion_event(event_output, pseudo_labels)
            loss_time = criterion_time(
                time_output.squeeze(), torch.zeros_like(time_output.squeeze())
            )  # Placeholder time labels

            # Calculate total loss and perform backpropagation
            total_loss = loss_event + loss_time
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(
            f"Self-training Epoch {epoch+1}, Loss: {running_loss/len(martian_data_loader)}"
        )


class SpectrogramCNN(nn.Module):
    """
    A CNN model identifying features from images of spectrograms
    """

    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        # CNN layers for spectrogram feature extraction
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # For grayscale spectrograms
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Add a placeholder for the fully connected input size
        self.fc_input_size = None  # To be calculated dynamically

        # Define fully connected layers (we'll initialize them later)
        self.fc1 = None
        self.fc_event = nn.Linear(128, 3)  # For event classification (3 classes)
        self.fc_time = nn.Linear(128, 1)  # For arrival time prediction

    def forward(self, x):
        """Forward propogation"""
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Initialize fully connected layers dynamically if not already initialized
        if self.fc1 is None:
            self.fc_input_size = x.size(1)
            self.fc1 = nn.Linear(self.fc_input_size, 128)

        x = torch.relu(self.fc1(x))

        # Separate heads for classification and regression
        event_output = self.fc_event(x)  # Output for event classification
        time_output = self.fc_time(x)  # Output for arrival time prediction

        return event_output, time_output

    def train_on_lunar_data(
        self,
        lunar_data_loader,
        criterion_event,
        criterion_time,
        optimizer,
        num_epochs=10,
    ):
        """Training on a labeled lunar dataset"""
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, event_labels, time_labels in lunar_data_loader:
                optimizer.zero_grad()
                event_output, time_output = self.forward(inputs)
                loss_event = criterion_event(event_output, event_labels)
                loss_time = criterion_time(time_output.squeeze(), time_labels)
                loss = loss_event + loss_time
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(lunar_data_loader)}")

    def train_on_martian_data(
        self,
        martian_data_loader,
        optimizer,
        num_epochs=10,
    ):
        """Training on a unlabeled martian dataset"""
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs in martian_data_loader:
                optimizer.zero_grad()

                # Forward pass through the model (get pseudo-labels)
                event_output, time_output = self.forward(inputs)
                _, pseudo_labels = torch.max(event_output, 1)

                # For now, we do not have ground truth for martian data, so we only optimize on pseudo-labels
                loss_event = nn.CrossEntropyLoss()(event_output, pseudo_labels)
                loss_time = nn.MSELoss()(
                    time_output, torch.zeros_like(time_output)
                )  # Zero as placeholder

                loss = loss_event + loss_time
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                f"Self-training Epoch {epoch+1}, Loss: {running_loss / len(martian_data_loader)}"
            )

    # Model Evaluation
    def evaluate_model(self, data_loader):
        self.eval()
        event_preds, time_preds, event_true, time_true = [], [], [], []
        with torch.no_grad():
            for images, event_labels, time_labels in data_loader:
                event_output, time_output = self(images)
                _, event_pred_classes = torch.max(event_output, 1)
                event_preds.extend(event_pred_classes.cpu().numpy())
                time_preds.extend(time_output.cpu().numpy())
                event_true.extend(event_labels.cpu().numpy())
                time_true.extend(time_labels.cpu().numpy())
        event_accuracy = accuracy_score(event_true, event_preds)
        time_mae = mean_absolute_error(time_true, time_preds)
        print(f"Validation Event Accuracy: {event_accuracy:.4f}")
        print(f"Validation Time MAE: {time_mae:.4f}")
        ConfusionMatrixDisplay.from_predictions(event_true, event_preds)
        plt.show()

    # Model Save Function
    def save_model(self, model_name="seismic_cnn_model"):
        model_architecture = {
            "conv_layers": [
                {
                    "in_channels": self.conv1.in_channels,
                    "out_channels": self.conv1.out_channels,
                    "kernel_size": self.conv1.kernel_size,
                },
                {
                    "in_channels": self.conv2.in_channels,
                    "out_channels": self.conv2.out_channels,
                    "kernel_size": self.conv2.kernel_size,
                },
            ],
            "fc_layers": [
                {
                    "in_features": self.fc_event.in_features,
                    "out_features": self.fc_event.out_features,
                },
                {
                    "in_features": self.fc_time.in_features,
                    "out_features": self.fc_time.out_features,
                },
            ],
        }
        with open(f"{model_name}_architecture.json", "w") as f:
            json.dump(model_architecture, f, indent=4)
        torch.save(self.state_dict(), f"{model_name}_weights.pth")
        torch.save(self, f"{model_name}_full.pth")
        print(f"Model saved to {model_name}_full.pth")
