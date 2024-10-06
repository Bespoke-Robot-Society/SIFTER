import os
import pandas as pd
from obspy import read
from scipy import signal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset


class ImageHandler:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def apply_bandpass_filter(self, trace, sampling_rate, freqmin=0.5, freqmax=3.0):
        try:
            sos = signal.butter(
                4, [freqmin, freqmax], btype="bandpass", fs=sampling_rate, output="sos"
            )
            return signal.sosfilt(sos, trace)
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            return trace  # Return the original trace if filtering fails

    def apply_sta_lta(self, trace, nsta, nlta):
        """
        Apply STA/LTA to the trace data to detect potential arrival times.
        """
        cft = classic_sta_lta(trace, nsta, nlta)
        return cft

    def detect_event(self, trace, nsta, nlta, threshold_on=3.5, threshold_off=1.0):
        """
        Detect seismic event using STA/LTA and return onset time of event.
        """
        cft = self.apply_sta_lta(trace, nsta, nlta)
        onsets = trigger_onset(cft, threshold_on, threshold_off)
        if len(onsets) > 0:
            return onsets[0][0]  # Return the first detected onset time
        else:
            return None

    def convert_rel_to_abs_time(self, relative_time, start_time):
        """
        Convert relative time (in seconds) to absolute time using the trace start time.
        Ensures that the relative time is converted to a regular Python integer or float.
        """
        return (start_time + timedelta(seconds=float(relative_time))).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )

    def convert_abs_to_rel_time(self, abs_times, reference_time):
        """
        Convert a list of absolute times to relative times (in seconds) based on the reference time.
        """
        rel_times = []
        for abs_time in abs_times:
            try:
                time_obj = datetime.strptime(abs_time, "%Y-%m-%dT%H:%M:%S.%f")
                rel_time = (time_obj - reference_time).total_seconds()
                rel_times.append(rel_time)
            except Exception as e:
                print(f"Error converting absolute time to relative: {e}")
                continue
        return rel_times

    def preprocess_data(
        self, file_path, filename, arrival_time_abs=None, combine_images=True
    ):
        """
        Preprocess the data and detect events using STA/LTA for initial arrival time.
        """
        try:
            st = read(file_path)
            tr = st[0]
            tr_data = tr.data
            tr_times = tr.times()
            sampling_rate = tr.stats.sampling_rate
            starttime = tr.stats.starttime.datetime

            # Apply bandpass filter to the trace data
            filtered_trace = self.apply_bandpass_filter(tr_data, sampling_rate)

            # Apply STA/LTA for arrival time detection
            arrival_time_rel = self.detect_event(
                filtered_trace,
                nsta=int(sampling_rate * 1),
                nlta=int(sampling_rate * 20),
            )
            if arrival_time_rel is not None:
                arrival_time_abs = self.convert_rel_to_abs_time(
                    arrival_time_rel, starttime
                )
                print(
                    f"Detected Event Arrival Time (relative): {arrival_time_rel} s, (absolute): {arrival_time_abs}"
                )

            # Generate spectrogram
            f, t, sxx = signal.spectrogram(filtered_trace, sampling_rate)

            # Create combined image
            fig = plt.figure(figsize=(12, 10)) if combine_images else None

            # Plot filtered trace
            ax1 = (
                plt.subplot(3, 1, 1)
                if combine_images
                else plt.figure(figsize=(8, 6)).add_subplot(111)
            )
            ax1.plot(tr_times, filtered_trace, label="Filtered Trace")
            ax1.set_ylabel("Velocity (m/s)")
            ax1.set_xlabel("Time (s)")
            ax1.set_title(f"Seismic Trace\nFile: {filename}")
            ax1.legend(loc="upper left")

            # Plot spectrogram
            ax2 = (
                plt.subplot(3, 1, 2)
                if combine_images
                else plt.figure(figsize=(8, 6)).add_subplot(111)
            )
            vals = ax2.pcolormesh(t, f, sxx, cmap="gray", shading="gouraud")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (s)")
            cbar = plt.colorbar(vals, ax=ax2, orientation="horizontal")
            cbar.set_label("Power ((m/s)^2/sqrt(Hz))")
            ax2.set_title("Spectrogram")

            # Save combined or separate images
            save_path = os.path.join(self.save_dir, f"{filename}_combined.png")
            if not os.path.isfile(save_path):
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                fig.tight_layout()
                plt.savefig(save_path, dpi=300)

            plt.close(fig)
            return save_path
        except Exception as e:
            print(f"Error preprocessing data for {filename}: {e}")
            return None

    def preprocess_lunar_data(self, catalog, data_dir, combine_images=True):
        """
        Preprocess lunar data and generate spectrograms and trace plots.
        """
        lunar_data, lunar_labels, lunar_arrival_times = [], [], []
        for idx, row in catalog.iterrows():
            filename = row["filename"] + ".mseed"
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                arrival_time_rel = row["time_rel(sec)"]
                arrival_time_abs = row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]

                # Preprocess data and generate spectrograms and trace plots
                spectrogram_image_path = self.preprocess_data(
                    file_path,
                    filename,
                    arrival_time_abs=arrival_time_abs,
                    combine_images=combine_images,
                )

                if spectrogram_image_path:
                    lunar_data.append(spectrogram_image_path)
                    lunar_labels.append(row["mq_type"])
                    lunar_arrival_times.append(arrival_time_abs)
            else:
                print(f"File {filename} not found.")
        return lunar_data, lunar_labels, lunar_arrival_times

    def preprocess_martian_data(self, data_dir, combine_images=True):
        """
        Preprocess Martian data and generate spectrograms and trace plots.
        Martian data does not include arrival time labels, so only spectrograms are generated.
        """
        martian_data = []  # Storing paths to spectrogram images
        for file in os.listdir(data_dir):
            if file.endswith(".mseed"):
                file_path = os.path.join(data_dir, file)

                # Preprocess data (Note: Martian data does not have labeled arrival times)
                spectrogram_image_path = self.preprocess_data(
                    file_path, file, combine_images=combine_images
                )

                if spectrogram_image_path:
                    martian_data.append(spectrogram_image_path)

        return martian_data, None  # No labels for Martian data
