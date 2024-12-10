import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram


class PeakDetector:
    def __init__(
        self, threshold=0, min_distance=1, prominence=None, width=None
    ):
        self.threshold = threshold
        self.min_distance = min_distance
        self.prominence = prominence
        self.width = width
        self.window_number = 0

    def detect_peaks(self, signal, plot_results=False):
        peaks = []
        peak_properties = {"heights": [], "prominences": [], "widths": []}

        # Step 1: Initial Peak Detection
        for i in range(1, len(signal) - 1):
            if (
                signal[i] > signal[i - 1] + self.threshold
                and signal[i] > signal[i + 1] + self.threshold
            ):
                peaks.append(i)

        # Step 2: Enforce minimum distance between peaks
        peaks = self._enforce_min_distance(peaks, self.min_distance)

        # Step 3: Calculate properties of the peaks
        for peak in peaks:
            height = signal[peak]
            prom = (
                self._calculate_prominence(signal, peak)
                if self.prominence is not None
                else None
            )
            w = self._calculate_width(signal, peak) if self.width is not None else None

            # Filter based on prominence and width
            if self.prominence is not None and prom < self.prominence:
                continue
            if self.width is not None and w < self.width:
                continue

            peak_properties["heights"].append(height)
            peak_properties["prominences"].append(prom)
            peak_properties["widths"].append(w)
        
        if plot_results:
            self._plot_peaks(signal=signal, peaks=peaks)
            self.window_number += 1


        return peaks, peak_properties

    def _enforce_min_distance(self, peaks, min_distance):
        if len(peaks) <= 1:
            return peaks
        peaks_filtered = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - peaks_filtered[-1] >= min_distance:
                peaks_filtered.append(peaks[i])
        return peaks_filtered

    def _calculate_prominence(self, signal, peak):
        left_base, right_base = peak, peak
        while left_base > 0 and signal[left_base] <= signal[left_base - 1]:
            left_base -= 1
        while (
            right_base < len(signal) - 1
            and signal[right_base] <= signal[right_base + 1]
        ):
            right_base += 1
        left_min = np.min(signal[left_base : peak + 1])
        right_min = np.min(signal[peak : right_base + 1])
        min_height = max(left_min, right_min)
        return signal[peak] - min_height

    def _calculate_width(self, signal, peak):
        peak_height = signal[peak]
        half_height = (peak_height + self.threshold) / 2
        left_idx = peak
        right_idx = peak
        while left_idx > 0 and signal[left_idx] > half_height:
            left_idx -= 1
        while right_idx < len(signal) - 1 and signal[right_idx] > half_height:
            right_idx += 1
        return right_idx - left_idx

    def _plot_peaks(self, signal, peaks):
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label="Signal")
        plt.plot(peaks, [signal[i] for i in peaks], "x", label="Peaks")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Window {self.window_number}")
        plt.savefig(f"plots/peaks/window_{self.window_number}_peaks.png")
        plt.close()
