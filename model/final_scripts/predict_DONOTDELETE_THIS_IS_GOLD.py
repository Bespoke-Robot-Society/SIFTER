import os
import torch
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

class STA_LTA_Processor:
    def __init__(self, sampling_rate=6.625):
        self.sampling_rate = sampling_rate  # Default from the notebook's miniseed file info

    def process(self, data, sta_len=120, lta_len=600):
        """
        Process the seismic data using STA/LTA trigger.
        Args:
            data (np.ndarray): The input seismic data array.
            sta_len (int): Short-term average window length in seconds.
            lta_len (int): Long-term average window length in seconds.
        Returns:
            cft (np.ndarray): The characteristic function (STA/LTA ratio).
        """
        df = self.sampling_rate  # Sampling rate in Hz
        cft = classic_sta_lta(data, int(sta_len * df), int(lta_len * df))  # Calculate STA/LTA
        return cft

class Predictor:
    def __init__(self, cnn_model_path, save_dir, catalog_png_dir, batch_size=32, sampling_rate=6.625):
        """
        Initialize the predictor class with CNN model and directories.
        Args:
            cnn_model_path (str): Path to the trained CNN model (PyTorch .pth file).
            save_dir (str): Directory where predictions and results will be saved.
            catalog_png_dir (str): Directory where plots will be saved.
            batch_size (int): Batch size for predictions.
            sampling_rate (float): Sampling rate of the seismic data.
        """
        self.cnn_model = torch.load(cnn_model_path, map_location='cpu')
        self.cnn_model.eval()  # Set model to evaluation mode
        self.save_dir = save_dir
        self.catalog_png_dir = catalog_png_dir
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate

        # Create directories if they don't exist
        if not os.path.exists(self.catalog_png_dir):
            os.makedirs(self.catalog_png_dir)
        
    def process_and_predict(self, test_data_dirs):
        """
        Process seismic data from multiple test directories, apply STA/LTA, and predict arrival times using the CNN model.
        Args:
            test_data_dirs (list): List of test directories containing .mseed files.
        """
        results = []
        processor = STA_LTA_Processor(sampling_rate=self.sampling_rate)

        for test_data_dir in test_data_dirs:
            print(f"Processing test data in folder: {test_data_dir}")

            # Iterate over all .mseed files in the directory
            for filename in os.listdir(test_data_dir):
                if filename.endswith('.mseed'):
                    mseed_file = os.path.join(test_data_dir, filename)
                    try:
                        # Read the .mseed file
                        st = read(mseed_file)
                        tr = st[0]
                        tr_data = tr.data
                        tr_times = tr.times()

                        # Apply STA/LTA processing
                        cft = processor.process(tr_data)

                        # Trigger on and off times
                        thr_on = 4  # Trigger threshold 'on'
                        thr_off = 1.5  # Trigger threshold 'off'
                        on_off = trigger_onset(cft, thr_on, thr_off)

                        if len(on_off) == 0:
                            print(f"No trigger found for {mseed_file}")
                            continue

                        # Predicted arrival time from the first trigger 'on'
                        predicted_time_rel = tr_times[on_off[0][0]]  # Get the relative time in seconds
                        predicted_time_abs = tr.stats.starttime + timedelta(seconds=predicted_time_rel)

                        print(f"Detected Event Arrival Time (relative): {predicted_time_rel:.2f} s, (absolute): {predicted_time_abs}")

                        # Save results
                        result = {
                            'filename': filename,
                            'time_abs': predicted_time_abs,
                            'time_rel': predicted_time_rel
                        }
                        results.append(result)

                        # Extract the base name of the .mseed file (without extension)
                        base_filename = os.path.splitext(filename)[0]

                        # Plotting the result with the .mseed filename as the plot filename
                        self.save_plot(tr_times, tr_data, predicted_time_rel, base_filename)

                    except Exception as e:
                        print(f"Error processing {mseed_file}: {e}")

        # Save predictions to CSV
        self.save_results(results)

    def save_plot(self, tr_times, tr_data, predicted_time_rel, actual_filename):
        """
        Save the plot of the seismic trace with marked arrival times using the .mseed filename as the .png filename.
        Args:
            tr_times (np.ndarray): Array of time values (in seconds).
            tr_data (np.ndarray): Array of seismic data (velocity or amplitude).
            predicted_time_rel (float): The predicted arrival time in seconds (relative).
            actual_filename (str): The original filename of the .mseed file (without extension).
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(tr_times, tr_data, label='Seismic Data')

            # Mark predicted arrival time
            ax.axvline(x=predicted_time_rel, color='red', linestyle='--', label=f'Predicted Arrival: {predicted_time_rel:.2f} s')

            ax.set_title(f'Predicted Arrival Time ({actual_filename})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.legend()

            fig.tight_layout()

            # Save the plot using the .mseed filename with a .png extension
            plot_filename = os.path.join(self.catalog_png_dir, f'{actual_filename}.png')
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Saved plot for {actual_filename} at {plot_filename}")

        except Exception as e:
            print(f"Error saving plot for {actual_filename}: {e}")

    def save_results(self, results):
        """
        Save the predicted arrival times to a CSV file.
        """
        results_df = pd.DataFrame(results)
        output_csv = os.path.join(self.save_dir, 'predictions_catalog.csv')
        results_df.to_csv(output_csv, index=False)
        print(f"Predicted arrival times saved to {output_csv}")

# Main execution
if __name__ == "__main__":
    # Define the path to the trained model
    cnn_model_path = 'martian_seismic_cnn_model_full.pth'
    
    # Define test directories
    test_data_dirs = [
        'data/mars/test',               # Martian test data
        'data/lunar/test/data/S12_GradeB',  # Lunar test set 1
        'data/lunar/test/data/S15_GradeA',  # Lunar test set 2
        'data/lunar/test/data/S15_GradeB',  # Lunar test set 3
        'data/lunar/test/data/S16_GradeA',  # Lunar test set 4
        'data/lunar/test/data/S16_GradeB'   # Lunar test set 5
    ]

    # Directory to save results
    save_dir = 'model/model_output'
    
    # Directory to save .png plots
    catalog_png_dir = 'model/catalog_png'

    # Instantiate the Predictor class with the model path and save directories
    predictor = Predictor(cnn_model_path=cnn_model_path, save_dir=save_dir, catalog_png_dir=catalog_png_dir, batch_size=32)
    
    # Run the prediction process on the defined test data directories
    predictor.process_and_predict(test_data_dirs=test_data_dirs)