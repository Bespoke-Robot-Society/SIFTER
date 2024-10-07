import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model.cnn_model import SpectrogramArrivalCNN
from preprocessing.imagehandler import ImageHandler
from model.model_trainer import ModelTrainer
from preprocessing.dataloader import DataLoaderHandler
from config import (
    LUNAR_CATALOG_PATH,
    LUNAR_DATA_DIR,
    LUNAR_SAVE_DIR,
    MARTIAN_DATA_DIR,
    MARTIAN_SAVE_DIR,
    SAVE_DIR,
    MODEL_FILENAME,
    MODEL_DICT_FILENAME,
    ONNX_MODEL_PATH,
)


def evaluate_and_get_metrics(trainer, test_loader):
    """
    Evaluate the model and compute metrics like MSE, MAE, and R-squared on the test set.
    """
    trainer.cnn_model.eval()  # Set to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch depending on the structure returned by the dataloader
            if isinstance(batch, (list, tuple)):
                inputs, time_labels = batch[0], batch[1]
            else:
                inputs = batch
                time_labels = None  # Adjust as per the actual data structure

            # Forward pass
            time_output = trainer.cnn_model(inputs)

            # Collect predictions and actual labels
            all_preds.extend(time_output.cpu().numpy())
            all_labels.extend(time_labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute regression metrics
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    # Print metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

    return mse, mae, r2


def main():
    # Load lunar catalog
    lunar_catalog = pd.read_csv(LUNAR_CATALOG_PATH)

    # Preprocess lunar data
    image_handler = ImageHandler(LUNAR_SAVE_DIR)
    lunar_data, lunar_labels, lunar_arrival_times = image_handler.preprocess_lunar_data(
        catalog=lunar_catalog, data_dir=LUNAR_DATA_DIR, combine_images=True
    )

    # Convert arrival times to relative time in seconds
    reference_time = pd.Timestamp("1970-01-01")
    lunar_arrival_times_in_seconds = image_handler.convert_abs_to_rel_time(
        lunar_arrival_times, reference_time
    )

    # Train-test split on lunar data
    lunar_data_train, lunar_data_test, lunar_times_train, lunar_times_test = (
        train_test_split(
            lunar_data, lunar_arrival_times_in_seconds, test_size=0.2, random_state=42
        )
    )

    # Normalize the arrival times
    scaler = MinMaxScaler()
    lunar_times_train_normalized = scaler.fit_transform(
        np.array(lunar_times_train).reshape(-1, 1)
    ).flatten()
    lunar_times_test_normalized = scaler.transform(
        np.array(lunar_times_test).reshape(-1, 1)
    ).flatten()

    # Prepare DataLoader for training and testing
    dataloader_handler = DataLoaderHandler(batch_size=32)
    lunar_train_loader = dataloader_handler.prepare_data_for_training(
        lunar_data_train, lunar_times_train_normalized
    )
    lunar_test_loader = dataloader_handler.prepare_data_for_training(
        lunar_data_test, lunar_times_test_normalized
    )

    # Initialize the model
    cnn_model = SpectrogramArrivalCNN()

    # Set up loss function and optimizer
    criterion_time = torch.nn.MSELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.00001)

    # Train the model on lunar data
    lunar_trainer = ModelTrainer(cnn_model, criterion_time, optimizer)
    lunar_trainer.train(lunar_train_loader, num_epochs=20)

    # Evaluate the model on the test set
    lunar_trainer.evaluate(lunar_test_loader)

    # Save the trained model
    model_path = f"{SAVE_DIR}/{MODEL_FILENAME}"
    model_path_dict = f"{SAVE_DIR}/{MODEL_DICT_FILENAME}"
    lunar_trainer.save_cnn_model(model_path)
    lunar_trainer.save_cnn_model_state_dict(model_path_dict)

    # Load the full pretrained lunar model
    loaded_cnn_model = torch.load(model_path)
    martian_trainer = ModelTrainer(loaded_cnn_model, criterion_time, optimizer)

    # Preprocess and self-train on Martian data
    image_handler = ImageHandler(MARTIAN_SAVE_DIR)
    martian_images, _ = image_handler.preprocess_martian_data(data_dir=MARTIAN_DATA_DIR)
    martian_data_loader = dataloader_handler.get_unlabeled_data_loader(martian_images)

    # Self-training on Martian data
    martian_trainer.self_train_on_martian_data(
        martian_data_loader, criterion_time=criterion_time, num_epochs=10
    )

    # Save the fine-tuned model after self-training on Martian data
    martian_trainer.save_cnn_model(model_path)
    martian_trainer.save_cnn_model_state_dict(model_path_dict)

    # Evaluate the model and compute metrics on the test set
    evaluate_and_get_metrics(martian_trainer, lunar_test_loader)

    print("Loading data for onnx model")
    cnn_model.save_pytorch_model_to_onnx(
        model_path_dict, ONNX_MODEL_PATH, MARTIAN_DATA_DIR, MARTIAN_SAVE_DIR
    )


if __name__ == "__main__":
    main()
