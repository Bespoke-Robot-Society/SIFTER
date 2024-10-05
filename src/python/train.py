from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from config import (
    LUNAR_CATALOG_PATH,
    MARTIAN_DATA_DIR,
    LUNAR_DATA_DIR,
    SAVE_DIR,
)
from utils.helpers import load_lunar_catalog, encode_labels_and_convert_time
from utils.preprocessing import (
    preprocess_and_validate_lunar_data,
    preprocess_and_validate_martian_data,
    prepare_lunar_data_for_training,
    prepare_data_for_training,
)
from model.cnn_spectrogram import SpectrogramCNN


def train_and_save_model():
    """
    Train and save the CNN model
    """
    # Load the lunar catalog
    print("Loading lunar catalog...")
    lunar_catalog = load_lunar_catalog(LUNAR_CATALOG_PATH)
    print(lunar_catalog)

    # # Preprocess and validate lunar data
    # print("Preprocessing and validating lunar data...")
    # lunar_data, lunar_labels, lunar_arrival_times = preprocess_and_validate_lunar_data(
    #     lunar_catalog, LUNAR_DATA_DIR, SAVE_DIR, combine_images=True
    # )

    # # Encode labels and convert arrival times to numeric values
    # print("Encoding labels and converting arrival times...")
    # lunar_labels_encoded, lunar_arrival_times_numeric = encode_labels_and_convert_time(
    #     lunar_labels, lunar_arrival_times
    # )

    # # Prepare DataLoader for lunar data training
    # print("Preparing DataLoader for lunar training and validation sets...")
    # lunar_data_loader = prepare_lunar_data_for_training(
    #     lunar_data, lunar_labels_encoded, lunar_arrival_times_numeric
    # )

    # # Initialize and train the model on lunar data
    # print("Initializing and training the model on lunar data...")
    # model = SpectrogramCNN()
    # criterion_event = nn.CrossEntropyLoss()
    # criterion_time = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # if lunar_data_loader:
    #     print("Training model on lunar data...")
    #     model.train_on_lunar_data(
    #         lunar_data_loader, criterion_event, criterion_time, optimizer
    #     )
    # else:
    #     print("Error: Training skipped due to invalid DataLoader.")
    #     return

    # # Prepare DataLoader for lunar data training
    # print("Preparing DataLoader for Martian data (self-training)...")
    # martian_data = preprocess_and_validate_martian_data(
    #     MARTIAN_DATA_DIR, SAVE_DIR, combine_images=True
    # )
    # martian_data_loader = prepare_data_for_training(
    #     martian_data,
    #     labels=[0] * len(martian_data),
    #     time_labels=[0] * len(martian_data),  # Placeholder labels
    # )

    # # Train model on martian data
    # if martian_data_loader:
    #     print("Self-training model on Martian data...")
    #     model.train_on_martian_data(model, martian_data_loader, optimizer)
    # else:
    #     print(
    #         "Error: Self-training skipped due to invalid DataLoader for Martian data."
    #     )

    # # Save the trained model
    # print("Saving the trained model...")
    # model.save_model()
