import os
import pandas as pd
import torch
from cnn_model import SpectrogramArrivalCNN  # Ensure you have the model class imported
from preprocessing import Preprocessing
from dataloader import DataLoaderHandler

class Testing:
    def __init__(self, cnn_model_path, save_dir, batch_size=32):
        # Load the model and keep everything on the CPU
        self.cnn_model = torch.load(cnn_model_path, map_location='cpu')
        self.cnn_model.eval()  # Set model to evaluation mode
        self.save_dir = save_dir
        self.batch_size = batch_size

    def iterate_and_predict(self, test_data_dirs):
        """
        Iterate over a list of test folders and return the predicted arrival times.
        """
        preprocessor = Preprocessing(self.save_dir)
        dataloader_handler = DataLoaderHandler(batch_size=self.batch_size)

        results = []

        for test_data_dir in test_data_dirs:
            print(f"Processing test data in folder: {test_data_dir}")

            # Load test data from the current folder
            test_images, _ = preprocessor.preprocess_martian_data(data_dir=test_data_dir, combine_images=False)

            if len(test_images) == 0:
                print(f"No valid images found in folder: {test_data_dir}")
                continue

            # Prepare DataLoader for the current folder
            test_loader = dataloader_handler.prepare_unlabeled_data_loader(test_images)

            # Collect predictions
            arrival_time_predictions = []
            for batch in test_loader:
                batch = batch[0]  # Keep everything on the CPU
                with torch.no_grad():
                    predicted_times = self.cnn_model(batch)  # Predict on the batch
                    arrival_time_predictions.extend(predicted_times.numpy())  # Convert to NumPy

            # Save the results for this folder
            folder_results = {
                'folder': test_data_dir,
                'arrival_times': arrival_time_predictions
            }
            results.append(folder_results)

        # Save the results to a CSV file
        self.save_results(results)

    def save_results(self, results):
        """
        Save the predicted arrival times to a CSV file.
        """
        rows = []
        for result in results:
            folder = result['folder']
            for i, arrival_time in enumerate(result['arrival_times']):
                rows.append({'folder': folder, 'file_index': i, 'predicted_arrival_time(s)': arrival_time})

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.save_dir, 'catalog.csv')
        df.to_csv(csv_path, index=False)
        print(f"Arrival time predictions saved to {csv_path}")
