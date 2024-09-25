import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak import PeakDetector
import pytest
from scipy.datasets import electrocardiogram
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plot_peaks(signal, custom_peaks, scipy_peaks=None):
    plt.figure(figsize=(12, 6))

    # Plot the signal
    plt.plot(signal, label="ECG Signal")

    # Plot peaks detected by custom algorithm
    plt.plot(custom_peaks, signal[custom_peaks], "x", label="Custom Peaks", color='red')

    # Plot peaks detected by scipy if provided
    if scipy_peaks is not None:
        plt.plot(scipy_peaks, signal[scipy_peaks], "o", label="Scipy Peaks", color='green')

    # Labels and legend
    plt.title("ECG Signal with Peaks Detected by Custom and Scipy Algorithms")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # pytest.main()

    # Use the ECG signal from the scipy package
    signal = electrocardiogram()

    # We'll analyze a portion of the signal to avoid too large data
    signal = signal[2000:4000]

    # Create an instance of PeakDetector
    detector = PeakDetector(signal=signal)

    # Detect peaks using the custom algorithm
    custom_peaks, custom_properties = detector.detect_peaks()

    # Detect peaks using scipy's find_peaks function
    scipy_peaks, _ = find_peaks(signal)

    # Plot the results for comparison
    plot_peaks(signal=signal, custom_peaks=custom_peaks, scipy_peaks=scipy_peaks)