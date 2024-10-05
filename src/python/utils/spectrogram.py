from scipy import signal
from obspy import read
import matplotlib.pyplot as plt
import os

from .helpers import convert_rel_to_abs_time, apply_bandpass_filter


class Spectrogram:
    def __init__(
        self, mseed_file, arrival_time_rel, save_dir, filename, combine_images=True
    ):
        """Get spectrogram images if they already exist, or create them if they dont"""
        self.mseed_file = mseed_file
        self.arrival_time_rel = arrival_time_rel
        self.save_dir = save_dir
        self.filename = filename
        self.combine_images = combine_images

    def get_spectrogram(self):
        if self.combine_images:
            image_path = os.path.join(self.save_dir, f"{self.filename}_combined.png")
            if os.path.isfile(image_path):
                return image_path
        else:
            # Save the trace plot
            trace_path = os.path.join(self.save_dir, f"{self.filename}_trace.png")
            spectrogram_path = os.path.join(
                self.save_dir, f"{self.filename}_spectrogram.png"
            )
            if os.path.isfile(trace_path) and os.path.isfile(trace_path):
                return (trace_path, spectrogram_path)

        return self.plot_and_save_trace_spectrogram()

    # Plot filtered trace and spectrogram, mark arrival times, and save images
    def plot_and_save_trace_spectrogram(self):
        # Read mseed file and extract the trace
        st = read(self.mseed_file)
        tr = st[0]
        tr_data = tr.data
        tr_times = tr.times()
        sampling_rate = tr.stats.sampling_rate
        starttime = tr.stats.starttime.datetime

        # Convert relative time to absolute if provided
        arrival_time_abs = (
            convert_rel_to_abs_time(starttime, self.arrival_time_rel)
            if self.arrival_time_rel
            else None
        )

        # Apply bandpass filter to the trace data
        filtered_trace = apply_bandpass_filter(tr_data, sampling_rate)

        # Generate spectrogram
        f, t, sxx = signal.spectrogram(filtered_trace, sampling_rate)

        # Create figure for plotting trace and spectrogram
        fig = plt.figure(figsize=(12, 10)) if self.combine_images else None

        # Plot filtered trace
        ax1 = (
            plt.subplot(3, 1, 1)
            if self.combine_images
            else plt.figure(figsize=(8, 6)).add_subplot(111)
        )
        ax1.plot(tr_times, filtered_trace, label="Filtered Trace")
        if self.arrival_time_rel:
            ax1.axvline(x=self.arrival_time_rel, color="red", label="Arrival Detection")
        ax1.set_xlim([min(tr_times), max(tr_times)])
        ax1.set_ylabel("Velocity (m/s)")
        ax1.set_xlabel("Time (s)")
        ax1.set_title(f"Filtered Seismic Trace\nArrival Time: {arrival_time_abs}")
        ax1.legend(loc="upper left")

        # Plot spectrogram
        ax2 = (
            plt.subplot(3, 1, 2)
            if self.combine_images
            else plt.figure(figsize=(8, 6)).add_subplot(111)
        )
        vals = ax2.pcolormesh(t, f, sxx, cmap="gray", shading="gouraud")
        if self.arrival_time_rel:
            ax2.axvline(x=self.arrival_time_rel, color="red")
        ax2.set_xlim([min(t), max(t)])
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        cbar = plt.colorbar(vals, ax=ax2, orientation="horizontal")
        cbar.set_label("Power ((m/s)^2/sqrt(Hz))")
        ax2.set_title("Spectrogram")

        # Save images: combined or separate
        if self.combine_images:
            save_path = os.path.join(self.save_dir, f"{self.filename}_combined.png")
            fig.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Saved combined image: {save_path}")
        else:
            # Save the trace plot
            trace_save_path = os.path.join(self.save_dir, f"{self.filename}_trace.png")
            ax1.figure.tight_layout()
            plt.savefig(trace_save_path, dpi=300)
            plt.close(ax1.figure)
            print(f"Saved trace image: {trace_save_path}")

            # Save the spectrogram plot
            spectrogram_save_path = os.path.join(
                self.save_dir, f"{self.filename}_spectrogram.png"
            )
            ax2.figure.tight_layout()
            plt.savefig(spectrogram_save_path, dpi=300)
            plt.close(ax2.figure)
            print(f"Saved spectrogram image: {spectrogram_save_path}")

            # Return the trace or spectrogram save path if not combined
            save_path = trace_save_path  # Adjust as needed to return the trace or spectrogram path
            # You can return both if needed, e.g., (trace_save_path, spectrogram_save_path)

        return save_path  # Return the path of the saved image(s)
