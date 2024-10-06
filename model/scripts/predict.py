import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from obspy import read
from datetime import timedelta


# Define the CNN model
class SpectrogramArrivalCNN(nn.Module):
    def __init__(self):
        super(SpectrogramArrivalCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
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


# Preprocessing function for mseed data
def preprocess_data(file_path, combine_images=False):
    st = read(file_path)
    tr = st[0]
    tr_data = tr.data
    sampling_rate = tr.stats.sampling_rate

    # Apply bandpass filter to the trace data
    sos = signal.butter(4, [0.5, 3.0], btype="bandpass", fs=sampling_rate, output="sos")
    filtered_trace = signal.sosfilt(sos, tr_data)

    # Generate spectrogram
    f, t, Sxx = signal.spectrogram(filtered_trace, sampling_rate)

    # Normalize the spectrogram
    Sxx = np.log1p(Sxx)  # Log normalization to stabilize values

    return Sxx


# Predict arrival times
def predict_arrival(cnn_model, preprocessed_data, device):
    with torch.no_grad():
        # Reshape the preprocessed data to match the expected input shape (batch, channels, height, width)
        data_tensor = (
            torch.tensor(preprocessed_data, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        predicted_time = cnn_model(data_tensor).item()  # Get predicted time in seconds
    return predicted_time


# Iterate over test folders and predict arrival times
def iterate_and_predict(cnn_model, test_data_dirs, save_dir, device):
    results = []
    for test_data_dir in test_data_dirs:
        print(f"Processing test data in folder: {test_data_dir}")

        # Iterate over all files in the folder
        for file in os.listdir(test_data_dir):
            if file.endswith(".mseed"):
                file_path = os.path.join(test_data_dir, file)
                try:
                    # Preprocess the .mseed file
                    Sxx = preprocess_data(file_path)

                    # Predict arrival time
                    predicted_arrival_time_rel = predict_arrival(cnn_model, Sxx, device)

                    # Convert to absolute time (assuming the start time is known)
                    tr = read(file_path)[0]
                    starttime = tr.stats.starttime.datetime
                    predicted_arrival_time_abs = starttime + timedelta(
                        seconds=predicted_arrival_time_rel
                    )

                    # Save results (include full file path)
                    results.append(
                        {
                            "folder": test_data_dir,
                            "file": file_path,  # Save the full file path
                            "predicted_arrival_time_relative(s)": predicted_arrival_time_rel,
                            "predicted_arrival_time_absolute": predicted_arrival_time_abs,
                        }
                    )

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    # Save results to CSV
    save_results(results, save_dir)


# Save predicted arrival times to CSV
def save_results(results, save_dir):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "catalog.csv"), index=False)
    print(f"Arrival time predictions saved to {os.path.join(save_dir, 'catalog.csv')}")


# Main execution
if __name__ == "__main__":
    # Model path
    cnn_model_path = "martian_seismic_cnn_model_finetuned.pth"

    # Device management (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model and move it to the appropriate device
    cnn_model = torch.load(cnn_model_path, map_location=device)
    cnn_model.eval()

    # Define test directories
    test_data_dirs = [
        "data/mars/test",
        "data/lunar/test/data/S12_GradeB",
        "data/lunar/test/data/S15_GradeA",
        "data/lunar/test/data/S15_GradeB",
        "data/lunar/test/data/S16_GradeA",
        "data/lunar/test/data/S16_GradeB",
    ]

    # Directory to save results
    save_dir = "model/model_output"

    # Iterate over the test data directories and predict arrival times
    iterate_and_predict(cnn_model, test_data_dirs, save_dir, device)
