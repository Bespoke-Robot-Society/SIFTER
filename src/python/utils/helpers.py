from datetime import datetime, timedelta
from scipy import signal
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from obspy import read
import matplotlib.pyplot as plt
import os


# Apply bandpass filter to seismic trace
def apply_bandpass_filter(trace, sampling_rate, freqmin=0.5, freqmax=3.0):
    sos = signal.butter(
        4, [freqmin, freqmax], btype="bandpass", fs=sampling_rate, output="sos"
    )
    return signal.sosfilt(sos, trace)


def convert_rel_to_abs_time(start_time, time_rel):
    """
    Convert relative time to absolute time using the trace start time.
    """
    return (start_time + timedelta(seconds=float(time_rel))).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )


def encode_labels_and_convert_time(lunar_labels, lunar_arrival_times):
    print("Encoding lunar labels into integers...")
    label_encoder = LabelEncoder()
    lunar_labels_encoded = label_encoder.fit_transform(lunar_labels)

    print("Converting arrival times to relative times (seconds)...")
    lunar_arrival_times_numeric = [
        (pd.to_datetime(time) - pd.to_datetime(lunar_arrival_times[0])).total_seconds()
        for time in lunar_arrival_times
    ]

    print(f"Numeric Lunar Arrival Times: {lunar_arrival_times_numeric[:5]}")

    return lunar_labels_encoded, lunar_arrival_times_numeric


def flatten_image_list(image_list):
    """
    Ensure image list is flat in case there are nested lists of image paths.
    """
    if isinstance(image_list, (list, tuple)) and any(
        isinstance(i, (list, tuple)) for i in image_list
    ):
        return [item for sublist in image_list for item in sublist]
    return image_list
