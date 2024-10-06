import torch
import torch.nn as nn
from utils.helpers import load_real_data


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


def save_pytorch_model_to_onnx(
    model, model_path_dict, onnx_model_path, data_dir, save_dir, opset_version=11
):
    """
    Save a PyTorch model to ONNX format using real input data.

    Args:
        model_path (str): Path to the PyTorch model (.pth file).
        onnx_model_path (str): Path to save the ONNX model (.onnx file).
        real_input (torch.Tensor): Real input data to use for model export.
        opset_version (int): ONNX opset version to use (default is 11).
    """
    # load real data
    real_input = load_real_data(data_dir, save_dir)

    # Export the model to ONNX format using real input
    torch.onnx.export(
        load_model(model, model_path_dict),  # The model to be exported
        real_input,  # Use real input data
        onnx_model_path,  # Path where the ONNX model will be saved
        export_params=True,  # Store the trained parameter weights inside the model
        opset_version=opset_version,  # ONNX version to use
        do_constant_folding=True,  # Simplify the model by folding constants
        input_names=["input"],  # Input name in the ONNX model
        output_names=["output"],  # Output name in the ONNX model
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Dynamic batch size support
    )

    print(f"Model successfully exported to {onnx_model_path}")


# Added function to load model and handle state dict
def load_model(model, state_dict_path):
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

    # Filter out unexpected keys
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Load the filtered state dict
    model.load_state_dict(state_dict, strict=False)
    return model
