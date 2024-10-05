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
        X_train, y_event_train, y_time_train
    )
    validation_spectrogram_data_loader = SpectrogramDataLoader(
        X_val, y_event_val, y_time_val
    )
    train_loader = training_spectrogram_data_loader.get_data_loader()
    val_loader = validation_spectrogram_data_loader.get_data_loader()

    if train_loader is not None and val_loader is not None:
        print("DataLoader successfully created for training.")
    else:
        print("Error: DataLoader creation failed.")

    return train_loader, val_loader


def initialize_and_train_model(train_loader):
    print("Initializing SpectrogramCNN model...")
    model = SpectrogramCNN()
    criterion_event = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if train_loader:
        print("Training model on lunar data...")
        model.train_on_lunar_data(
            train_loader, criterion_event, criterion_time, optimizer
        )
    else:
        print("Error: Training skipped due to invalid DataLoader.")

    return model


def train_and_save_model():
    """
    Train and save the CNN model
    """
    # Process and validate date
    imageProcessor = ImageProcessor()
    lunar_data, lunar_labels, lunar_arrival_times = (
        imageProcessor.preprocess_and_validate_lunar_data()
    )
    martian_data, martian_arrival_times = (
        imageProcessor.preprocess_and_validate_martian_data()
    )

    # Encode labels and convert arrival times to numeric values
    print("Encoding labels and converting arrival times...")
    lunar_labels_encoded, lunar_arrival_times_numeric = encode_labels_and_convert_time(
        lunar_labels, lunar_arrival_times
    )

    # Prepare DataLoader for lunar data training
    train_loader, val_loader = prepare_lunar_data_for_training(
        lunar_data, lunar_labels_encoded, lunar_arrival_times_numeric
    )

    # Initialize and train the model on lunar data
    print("Initializing and training the model on lunar data...")
    model = initialize_and_train_model(train_loader)
