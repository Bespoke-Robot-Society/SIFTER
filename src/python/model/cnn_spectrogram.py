import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json


class SpectrogramCNN(nn.Module):
    """
    A CNN model identifying features from images of spectrograms
    """

    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, 128)  # edit to None if it doesn't work
        self.fc_event = nn.Linear(128, 3)
        self.fc_time = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)
        x = torch.relu(self.fc1(x))
        return self.fc_event(x), self.fc_time(x)

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
        criterion_event,
        criterion_time,
        optimizer,
        num_epochs=10,
    ):
        """Training on a unlabeled martian dataset"""
        self.train()  # Set the model to training mode
        print(len(martian_data_loader.dataset))
        for epoch in range(num_epochs):  # Loop over epochs
            running_loss = 0.0  # Track loss for current epoch
            for batch in martian_data_loader:
                images = batch[0]  # Extract image data from the batch
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass through the model to get predictions
                event_output, time_output = self.forward(images)

                # Generate pseudo-labels: assign the highest predicted class as the label
                _, pseudo_labels = torch.max(event_output, 1)

                # Calculate losses using pseudo-labels for events and zero as placeholder for time
                loss_event = criterion_event(event_output, pseudo_labels)
                loss_time = criterion_time(
                    time_output.squeeze(), torch.zeros_like(time_output.squeeze())
                )

                # Calculate the total loss and perform backpropagation
                total_loss = loss_event + loss_time
                total_loss.backward()  # Backpropagate the gradients
                optimizer.step()  # Update model parameters

                running_loss += total_loss.item()  # Accumulate the running loss
            # Print the average loss for the epoch
            print(
                f"Self-training Epoch {epoch+1}, Loss: {running_loss/len(martian_data_loader)}"
            )

    # Model Evaluation
    def evaluate_model(self, data_loader):
        self.eval()  # Set the model to evaluation mode
        event_preds, time_preds, event_true, time_true = (
            [],
            [],
            [],
            [],
        )  # Lists to hold predictions and ground truth

        # No need to track gradients during evaluation
        with torch.no_grad():
            for (
                images,
                event_labels,
                time_labels,
            ) in data_loader:  # Loop over the validation data
                event_output, time_output = self.forward(
                    images
                )  # Forward pass to get outputs

                # Get the predicted event classes and append to lists
                _, event_pred_classes = torch.max(event_output, 1)
                event_preds.extend(
                    event_pred_classes.cpu().numpy()
                )  # Convert to numpy for easy manipulation
                time_preds.extend(time_output.cpu().numpy())  # Store predicted times
                event_true.extend(event_labels.cpu().numpy())  # Store true event labels
                time_true.extend(time_labels.cpu().numpy())  # Store true times

        # Calculate accuracy and mean absolute error (MAE) for event prediction and time prediction
        event_accuracy = accuracy_score(event_true, event_preds)
        time_mae = mean_absolute_error(time_true, time_preds)

        # Print evaluation results
        print(f"Validation Event Accuracy: {event_accuracy:.4f}")
        print(f"Validation Time MAE: {time_mae:.4f}")

        # Plot confusion matrix for event classification
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
