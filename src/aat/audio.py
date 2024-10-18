import numpy as np

class AudioWaveform():
    def __init__(self, waveform, sampling_rate):
        self.waveform: np.ndarray = waveform
        self.sampling_rate = sampling_rate

        self.duration_seconds: float = self.waveform.shape[-1] / self.sampling_rate

        return

    def assert_sampling_rate(self, expected_sapmling_rate: int):
        assert self.sampling_rate == expected_sapmling_rate, f"Audio sampling rate mismatch: ausio_sampling_rate={self.sampling_rate}, expected_sapmling_rate={expected_sapmling_rate}"
