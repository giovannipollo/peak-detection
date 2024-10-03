class PeakDetectorSingleSample:
    """
    A class for real-time peak detection in streaming data using basic Python constructs.

    This class implements an online algorithm for detecting peaks in a continuous
    stream of data samples. It maintains a fixed-size buffer of recent samples
    and applies various criteria to identify and confirm peaks.

    Attributes:
        buffer_size (int): The number of recent samples to keep in the buffer.
        threshold (float): The minimum height difference for a sample to be considered a peak.
        min_distance (int): The minimum number of samples between peaks.
        prominence (float): The minimum prominence for a peak to be confirmed.
        width (int): The minimum width for a peak to be confirmed.
    """

    def __init__(self, buffer_size=100, threshold=0, min_distance=1, prominence=None, width=None):
        """
        Initialize the BasicStreamingPeakDetector.

        Args:
            buffer_size (int): Size of the buffer for recent samples.
            threshold (float): Minimum height difference for peak detection.
            min_distance (int): Minimum distance between peaks.
            prominence (float): Minimum prominence for a peak to be confirmed.
            width (int): Minimum width for a peak to be confirmed.
        """
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.min_distance = min_distance
        self.prominence = prominence
        self.width = width
        
        self.buffer = [0] * buffer_size  # Fixed-size buffer initialized with zeros
        self.buffer_index = 0  # Current position in the buffer
        self.potential_peaks = []  # List to store potential peaks
        self.confirmed_peaks = []  # List to store confirmed peaks
        self.sample_count = 0  # Counter for the total number of samples processed

    def add_sample(self, sample):
        """
        Process a new sample, updating peak detection.

        This method adds a new sample to the buffer, checks for potential peaks,
        and processes confirmed peaks when the buffer is full.

        Args:
            sample (float): The new data sample to process.

        Returns:
            list: The current list of confirmed peaks.
        """
        self.sample_count += 1
        
        # Add the new sample to the buffer
        self.buffer[self.buffer_index] = sample
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        # Check for potential peak
        if self.sample_count >= 3:
            self._check_potential_peak()

        # Process potential peaks
        if self.sample_count >= self.buffer_size:
            self._process_potential_peaks()

        return self.confirmed_peaks

    def _check_potential_peak(self):
        """
        Check if the second-to-last sample in the buffer is a potential peak.

        A sample is considered a potential peak if it's higher than both its
        neighbors by at least the threshold value.
        """
        current_index = (self.buffer_index - 2) % self.buffer_size
        prev_index = (current_index - 1) % self.buffer_size
        next_index = (current_index + 1) % self.buffer_size

        if (self.buffer[current_index] > self.buffer[prev_index] + self.threshold and
            self.buffer[current_index] > self.buffer[next_index] + self.threshold):
            self.potential_peaks.append((self.sample_count - 2, self.buffer[current_index]))

    def _process_potential_peaks(self):
        """
        Process potential peaks to confirm or reject them.

        This method checks potential peaks that are now in the middle of the buffer,
        applying additional criteria (min_distance, prominence, width) to confirm
        them as actual peaks.
        """
        new_potential_peaks = []
        for peak in self.potential_peaks:
            if self.sample_count - peak[0] >= self.buffer_size // 2:
                if self._check_peak_validity(peak):
                    self.confirmed_peaks.append(peak)
            else:
                new_potential_peaks.append(peak)
        self.potential_peaks = new_potential_peaks

    def _check_peak_validity(self, peak):
        """
        Check if a potential peak meets all criteria to be confirmed.

        Args:
            peak (tuple): A tuple containing (peak_position, peak_value).

        Returns:
            bool: True if the peak is valid, False otherwise.
        """
        # Check minimum distance criterion
        if self.min_distance:
            if self.confirmed_peaks and peak[0] - self.confirmed_peaks[-1][0] < self.min_distance:
                return False

        # Check prominence criterion
        if self.prominence:
            if self._calculate_prominence(peak) < self.prominence:
                return False

        # Check width criterion
        if self.width:
            if self._calculate_width(peak) < self.width:
                return False

        return True

    def _calculate_prominence(self, peak):
        """
        Calculate the prominence of a peak.

        Prominence is the vertical distance between the peak and its lowest contour line.

        Args:
            peak (tuple): A tuple containing (peak_position, peak_value).

        Returns:
            float: The calculated prominence of the peak.
        """
        peak_index = peak[0] % self.buffer_size
        left_min = min(self.buffer[peak_index:] + self.buffer[:peak_index])
        right_min = min(self.buffer[peak_index:] + self.buffer[:peak_index])
        return peak[1] - max(left_min, right_min)

    def _calculate_width(self, peak):
        """
        Calculate the width of a peak at half prominence.

        Args:
            peak (tuple): A tuple containing (peak_position, peak_value).

        Returns:
            int: The calculated width of the peak.
        """
        half_height = (peak[1] + self.threshold) / 2
        peak_index = peak[0] % self.buffer_size
        left = right = peak_index

        while self.buffer[left] > half_height:
            left = (left - 1) % self.buffer_size
            if left == peak_index:
                break

        while self.buffer[right] > half_height:
            right = (right + 1) % self.buffer_size
            if right == peak_index:
                break

        if right > left:
            return right - left
        else:
            return self.buffer_size - left + right

    def get_peaks(self):
        """
        Get the list of confirmed peaks.

        Returns:
            list: A list of tuples, each containing (peak_position, peak_value).
        """
        return self.confirmed_peaks