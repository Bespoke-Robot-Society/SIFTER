import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import Preprocessing
from cnn_model import SpectrogramArrivalCNN
from training import ModelTrainer
from dataloader import DataLoaderHandler
from testing import Testing

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
    # Paths to your data
    lunar_catalog_path = 'data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
    lunar_data_directory = 'data/lunar/training/data/S12_GradeA/'
    lunar_data_images_dir = 'model/model_output/lunar_preprocessed_images/'
    save_dir_lunar = lunar_data_images_dir

    martian_data_directory = 'data/mars/training/data/'
    martian_data_images_dir = 'model/model_output/martian_preprocessed_images/'
    save_dir_mars = martian_data_images_dir

    # Load lunar catalog
    lunar_catalog = pd.read_csv(lunar_catalog_path)

    # Preprocess lunar data
    preprocessor = Preprocessing(save_dir_lunar)
    lunar_data, lunar_labels, lunar_arrival_times = preprocessor.preprocess_lunar_data(
        catalog=lunar_catalog, data_dir=lunar_data_directory, combine_images=True
    )

    # Convert arrival times to relative time in seconds
    reference_time = pd.Timestamp('1970-01-01')
    lunar_arrival_times_in_seconds = preprocessor.convert_abs_to_rel_time(lunar_arrival_times, reference_time)

    # Train-test split on lunar data
    lunar_data_train, lunar_data_test, lunar_times_train, lunar_times_test = train_test_split(
        lunar_data, lunar_arrival_times_in_seconds, test_size=0.2, random_state=42
    )

    # Normalize the arrival times
    scaler = MinMaxScaler()
    lunar_times_train_normalized = scaler.fit_transform(np.array(lunar_times_train).reshape(-1, 1)).flatten()
    lunar_times_test_normalized = scaler.transform(np.array(lunar_times_test).reshape(-1, 1)).flatten()

    # Prepare DataLoader for training and testing
    dataloader_handler = DataLoaderHandler(batch_size=32)
    lunar_train_loader = dataloader_handler.prepare_data_for_training(lunar_data_train, lunar_times_train_normalized)
    lunar_test_loader = dataloader_handler.prepare_data_for_training(lunar_data_test, lunar_times_test_normalized)

    # Initialize the model
    cnn_model = SpectrogramArrivalCNN()

    # Set up loss function and optimizer
    criterion_time = torch.nn.MSELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.00001)

    # Train the model on lunar data
    trainer = ModelTrainer(cnn_model, criterion_time, optimizer)
    trainer.train(lunar_train_loader, num_epochs=20)

    # Evaluate the model on the test set
    trainer.evaluate(lunar_test_loader)

    # Save the trained model
    trainer.save_cnn_model('lunar_seismic_cnn_model_full.pth')
    trainer.save_cnn_model_state_dict('lunar_seismic_cnn_model_state_dict.pth')

    # Load the full pretrained lunar model
    cnn_model = torch.load('lunar_seismic_cnn_model_full.pth')
    cnn_model.train()  # Set to training mode

    # Preprocess and self-train on Martian data
    preprocessor = Preprocessing(save_dir_mars)
    martian_images, _ = preprocessor.preprocess_martian_data(data_dir=martian_data_directory)
    martian_data_loader = dataloader_handler.prepare_unlabeled_data_loader(martian_images)

    # Self-training on Martian data
    trainer.self_train_on_martian_data(martian_data_loader, criterion_time=criterion_time, num_epochs=10)

    # Save the fine-tuned model after self-training on Martian data
    trainer.save_cnn_model('martian_seismic_cnn_model_full.pth')
    trainer.save_cnn_model_state_dict('martian_seismic_cnn_model_state_dict.pth')

    # Evaluate the model and compute metrics on the test set
    mse, mae, r2 = evaluate_and_get_metrics(trainer, lunar_test_loader)

if __name__ == "__main__":
    main()
