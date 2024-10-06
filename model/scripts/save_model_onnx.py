import torch
import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
from cnn.cnn_model import SpectrogramArrivalCNN  # Import your model
from preprocessing.imagehandler import ImageHandler
import numpy as np


def load_real_data(data_dir, batch_size=1):
    """
    Loads real data from your dataset to be used as input for exporting the model.
    This assumes the data is in a directory and has been preprocessed to spectrogram format.

    Args:
        data_dir (str): Path to the data directory.
        batch_size (int): The batch size for the DataLoader (default is 1).

    Returns:
        torch.Tensor: A single batch of real input data.
    """
    # Preprocess the data using your Preprocessing class
    preprocessor = ImageHandler(
        save_dir=None
    )  # Update this with actual save_dir if needed
    spectrogram_data, _ = preprocessor.preprocess_martian_data(
        data_dir
    )  # Preprocess your data

    # Create DataLoader for real data
    real_data = [torch.tensor(sxx, dtype=torch.float32) for sxx in spectrogram_data]
    real_data = torch.stack(real_data)  # Stack tensors into a batch

    # Assuming the data needs to be 4D (batch_size, channels, height, width)
    real_data = real_data.unsqueeze(
        1
    )  # Add the channel dimension for a single channel (1, H, W)

    dataset = TensorDataset(real_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load one batch of real data
    (real_input,) = next(iter(data_loader))  # Extract the first batch (inputs only)

    return real_input


def save_pytorch_model_to_onnx(
    model_path, onnx_model_path, real_input, opset_version=11
):
    """
    Save a PyTorch model to ONNX format using real input data.

    Args:
        model_path (str): Path to the PyTorch model (.pth file).
        onnx_model_path (str): Path to save the ONNX model (.onnx file).
        real_input (torch.Tensor): Real input data to use for model export.
        opset_version (int): ONNX opset version to use (default is 11).
    """
    # Load the PyTorch model
    model = SpectrogramArrivalCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Export the model to ONNX format using real input
    torch.onnx.export(
        model,  # The model to be exported
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


if __name__ == "__main__":
    # Path to the saved PyTorch model (.pth file)
    pytorch_model_path = "martian_seismic_cnn_model_finetuned.pth"

    # Path where you want to save the ONNX model
    onnx_model_path = "martian_seismic_cnn_model.onnx"

    # Load real data for export
    data_directory = "data/mars/training/data/"  # Update with your data directory path
    real_input = load_real_data(data_directory, batch_size=1)

    # Save the model to ONNX format using real data
    save_pytorch_model_to_onnx(pytorch_model_path, onnx_model_path, real_input)
