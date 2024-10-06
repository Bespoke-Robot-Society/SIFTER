from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from config import (
    LUNAR_CATALOG_PATH,
    MARTIAN_DATA_DIR,
    LUNAR_DATA_DIR,
)
from utils.helpers import encode_labels_and_convert_time
from utils.dataloader import SpectrogramDataLoader
from utils.image_processor import ImageProcessor
from model.cnn_spectrogram import SpectrogramCNN


def prepare_lunar_data_for_training(
    lunar_data, lunar_labels_encoded, lunar_arrival_times_numeric
):
    print("Splitting lunar data into training and validation sets...")
    X_train, X_val, y_event_train, y_event_val, y_time_train, y_time_val = (
        train_test_split(
            lunar_data,
            lunar_labels_encoded,
            lunar_arrival_times_numeric,
            test_size=0.2,
            random_state=42,
        )
    )

    print("Preparing DataLoader for lunar training data...")
    training_spectrogram_data_loader = SpectrogramDataLoader(
        X_train,
        y_time_train,
        y_event_train,
    )
    validation_spectrogram_data_loader = SpectrogramDataLoader(
        X_val,
        y_time_val,
        y_event_val,
    )
    train_loader = training_spectrogram_data_loader.get_data_loader()
    val_loader = validation_spectrogram_data_loader.get_data_loader()

    if train_loader is not None and val_loader is not None:
        print("Lunar DataLoader successfully created for training.")
    else:
        print("Error: DataLoader creation failed.")

    return train_loader, val_loader


def prepare_martian_data_for_training(martian_data, martian_arrival_times):
    print("Preparing DataLoader for Martian data (self-training)...")
    training_spectrogram_data_loader = SpectrogramDataLoader(
        martian_data,
        time_labels=martian_arrival_times,
    )  # Placeholder labels
    train_loader = training_spectrogram_data_loader.get_unlabeled_data_loader()

    if train_loader is not None:
        print("Martian DataLoader successfully created for training.")
    else:
        print(
            "Error: Self-training skipped due to invalid DataLoader for Martian data."
        )

    return train_loader


def train_and_save_model():
    """
    Train and save the CNN model
    """
    # Process and validate date
    imageProcessor = ImageProcessor()
    print("Preprocessing and validating Lunar data...")
    lunar_data = imageProcessor.preprocess_and_validate_lunar_data()
    print("Preprocessing and validating Martian data...")
    martian_data = imageProcessor.preprocess_and_validate_martian_data()

    # Prepare DataLoader for training
    lunar_labels_encoded = [0] * len(lunar_data)  # Placeholder labels
    lunar_arrival_times_numeric = [0] * len(lunar_data)  # Placeholder arrival times
    martian_arrival_times_numeric = [0] * len(martian_data)  # Placeholder arrival times
    lunar_train_loader, lunar_val_loader = prepare_lunar_data_for_training(
        lunar_data, lunar_labels_encoded, lunar_arrival_times_numeric
    )
    martian_train_loader = prepare_martian_data_for_training(
        martian_data, martian_arrival_times_numeric
    )

    print("Initializing SpectrogramCNN model...")
    model = SpectrogramCNN()
    criterion_event = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if lunar_train_loader:
        print("Training model on lunar data...")
        model.train_on_lunar_data(
            lunar_train_loader, criterion_event, criterion_time, optimizer
        )
    else:
        print("Error: Training skipped due to invalid Lunar DataLoader.")

    if martian_train_loader:
        print("Training model on martian data...")
        model.train_on_martian_data(
            martian_train_loader, criterion_event, criterion_time, optimizer
        )
    else:
        print("Error: Training skipped due to invalid Martian DataLoader.")
