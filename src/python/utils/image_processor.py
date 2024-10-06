from .spectrogram import Spectrogram
import os
import pandas as pd
from config import (
    LUNAR_CATALOG_PATH,
    MARTIAN_DATA_DIR,
    LUNAR_DATA_DIR,
    LUNAR_DATA_IMAGES_DIR,
    MARTIAL_DATA_IMAGES_DIR,
)


class ImageProcessor:
    """Class to create and process spectrogram images from raw lunar and martian data"""

    def __init__(self):
        print(f"Loading lunar catalog from: {LUNAR_CATALOG_PATH}")
        self.lunar_catalog = pd.read_csv(LUNAR_CATALOG_PATH)

    def get_spectrogram_images_lunar(self, combine_images=True):
        if not os.path.exists(LUNAR_DATA_IMAGES_DIR):
            os.makedirs(LUNAR_DATA_IMAGES_DIR)

        lunar_data = []

        print(f"Processing Lunar Data: {len(self.lunar_catalog)} records found.")

        for idx, row in self.lunar_catalog.iterrows():
            filename = row["filename"] + ".mseed"
            file_path = os.path.join(LUNAR_DATA_DIR, filename)

            if os.path.exists(file_path):
                # Extract time_rel and time_abs
                arrival_time_rel = row["time_rel(sec)"]
                arrival_time_abs = row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]

                # Debug logging
                print(f"\nProcessing file: {filename}")
                print(f"Arrival Time (rel): {arrival_time_rel} seconds")
                print(f"Arrival Time (abs): {arrival_time_abs}")

                # Generate and save spectrogram images
                spectrogram = Spectrogram(
                    file_path,
                    arrival_time_rel,
                    LUNAR_DATA_DIR,
                    filename,
                    combine_images,
                )
                spectrogram_image_path = spectrogram.get_spectrogram()

                # Append the image path (not .mseed path) to lunar_data
                lunar_data.append(
                    spectrogram_image_path
                )  # This should now contain the path to the saved image

            else:
                print(f"File {filename} not found.")

        # Ensure that lunar_data contains paths to saved images
        return lunar_data

    def get_spectrogram_images_mars(self, combine_images=True):
        if not os.path.exists(MARTIAL_DATA_IMAGES_DIR):
            os.makedirs(MARTIAL_DATA_IMAGES_DIR)

        martian_images = []

        # List all .mseed files in the data directory
        mseed_files = [f for f in os.listdir(MARTIAN_DATA_DIR) if f.endswith(".mseed")]

        if len(mseed_files) == 0:
            print("No .mseed files found in the directory.")
            return martian_images

        # Iterate over each .mseed file
        for filename in mseed_files:
            file_path = os.path.join(MARTIAN_DATA_DIR, filename)

            # Derive the corresponding CSV file path from the .mseed filename
            csv_file_name = filename.replace(".mseed", ".csv")
            csv_file_path = os.path.join(MARTIAN_DATA_DIR, csv_file_name)

            # Ensure the CSV file exists before processing
            if not os.path.exists(csv_file_path):
                print(f"CSV file not found for {filename}: {csv_file_path}")
                continue

            # Read the CSV file to get the rel_time (relative time in seconds)
            try:
                csv_data = pd.read_csv(csv_file_path)
                if "rel_time(sec)" not in csv_data.columns:
                    print(f"CSV does not contain 'rel_time(sec)': {csv_file_path}")
                    continue
                arrival_time_rel = csv_data["rel_time(sec)"].iloc[0]
            except Exception as e:
                print(f"Error reading CSV file {csv_file_path}: {e}")
                continue

            # Process and generate spectrogram
            try:
                spectrogram = Spectrogram(
                    file_path,
                    arrival_time_rel,
                    MARTIAL_DATA_IMAGES_DIR,
                    filename,
                    combine_images,
                )
                image_path = spectrogram.get_spectrogram()
                martian_images.append(image_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return martian_images

    def preprocess_and_validate_lunar_data(self, combine_images=True):
        print("Preprocessing lunar data...")
        lunar_data = self.get_spectrogram_images_lunar(combine_images)

        if len(lunar_data) > 0:
            for i in range(min(5, len(lunar_data))):
                img_path = lunar_data[i]
                if os.path.exists(img_path) and img_path.endswith(".png"):
                    print(f"Valid image path: {img_path}")
                else:
                    print(f"Invalid image path or file does not exist: {img_path}")
        else:
            print("Error: No lunar image data found.")

        return lunar_data

    def preprocess_and_validate_martian_data(self, combine_images=True):
        print("Preprocessing Martian data (no labels)...")
        martian_data = self.get_spectrogram_images_mars(combine_images)
        if len(martian_data) > 0:
            print(f"Martian Data: {len(martian_data)} files found.")
            for i in range(min(5, len(martian_data))):
                img_path = martian_data[i]
                if os.path.exists(img_path) and img_path.endswith(".png"):
                    print(f"Valid Martian image path: {img_path}")
                else:
                    print(
                        f"Invalid Martian image path or file does not exist: {img_path}"
                    )
        else:
            print("Error: No Martian data found.")

        return martian_data
