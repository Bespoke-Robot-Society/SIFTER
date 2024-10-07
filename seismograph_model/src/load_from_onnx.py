from onnxruntime import InferenceSession
from config import ONNX_MODEL_PATH, TEST_DATA_DIRS
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model.cnn_model import SpectrogramArrivalCNN
from preprocessing.imagehandler import ImageHandler
from model.model_trainer import ModelTrainer
from preprocessing.dataloader import DataLoaderHandler, transform
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
    CATALOG_PNG_DIR,
)
import os
import torch
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image
from model.model_predictor import STA_LTA_Processor


def save_plot(tr_times, tr_data, predicted_time_rel, actual_filename):
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
        ax.plot(tr_times, tr_data, label="Seismic Data")

        # Mark predicted arrival time
        ax.axvline(
            x=predicted_time_rel,
            color="red",
            linestyle="--",
            label=f"Predicted Arrival: {predicted_time_rel:.2f} s",
        )

        ax.set_title(f"Predicted Arrival Time ({actual_filename})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()

        fig.tight_layout()

        # Save the plot using the .mseed filename with a .png extension
        plot_filename = os.path.join(CATALOG_PNG_DIR, f"{actual_filename}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"Saved plot for {actual_filename} at {plot_filename}")

    except Exception as e:
        print(f"Error saving plot for {actual_filename}: {e}")


def load_and_run():
    csv_results = []
    json_results = []
    sess = InferenceSession(ONNX_MODEL_PATH)
    image_handler = ImageHandler(CATALOG_PNG_DIR)
    dataloader_handler = DataLoaderHandler()
    processor = STA_LTA_Processor(sampling_rate=6.625)

    for test_data_dir in TEST_DATA_DIRS:
        print(f"Processing test data in folder: {test_data_dir}")

        # Iterate over all .mseed files in the directory
        for filename in os.listdir(test_data_dir):
            if filename.endswith(".mseed"):
                mseed_file = os.path.join(test_data_dir, filename)
                try:
                    # Read the .mseed file
                    base_filename = os.path.splitext(filename)[0]
                    st = read(mseed_file)
                    tr = st[0]
                    tr_data = tr.data
                    tr_times = tr.times()

                    spectrogram_image_path = image_handler.preprocess_data(
                        mseed_file, base_filename
                    )
                    img_tensor = transform(Image.open(spectrogram_image_path))
                    predicted_time_cnn_rel = float(
                        str(sess.run(None, {"input": [img_tensor]})[0][0][0])
                    )
                    predicted_time_cnn_abs = tr.stats.starttime + timedelta(
                        seconds=predicted_time_cnn_rel
                    )

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
                    predicted_time_rel_lta = tr_times[
                        on_off[0][0]
                    ]  # Get the relative time in seconds
                    predicted_time_abs_lta = tr.stats.starttime + timedelta(
                        seconds=predicted_time_rel_lta
                    )

                    print(
                        f"LTA/STA - Detected Event Arrival Time (relative): {predicted_time_rel_lta:.2f} s, (absolute): {predicted_time_abs_lta}"
                    )
                    print(
                        f"CNN - Detected Event Arrival Time (relative): {predicted_time_cnn_rel:.2f} s, (absolute): {predicted_time_cnn_abs}"
                    )

                    # Extract the base name of the .mseed file (without extension)
                    base_filename = os.path.splitext(filename)[0]
                    print(base_filename)

                    # Plotting the result with the .mseed filename as the plot filename
                    save_plot(tr_times, tr_data, predicted_time_cnn_rel, base_filename)

                except Exception as e:
                    print(f"Error processing {mseed_file}: {e}")


load_and_run()
