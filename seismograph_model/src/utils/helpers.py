import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
from preprocessing.imagehandler import ImageHandler
from preprocessing.dataloader import DataLoaderHandler


def load_real_data(data_dir, save_dir, batch_size=1):
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
    dataloader_handler = DataLoaderHandler(batch_size=32)
    image_handler = ImageHandler(
        save_dir=save_dir
    )  # Update this with actual save_dir if needed
    spectrogram_data, _ = image_handler.preprocess_martian_data(
        data_dir
    )  # Preprocess your data
    data_loader = dataloader_handler.get_unlabeled_data_loader(spectrogram_data)

    # Load one batch of real data
    (real_input,) = next(iter(data_loader))  # Extract the first batch (inputs only)

    return real_input
