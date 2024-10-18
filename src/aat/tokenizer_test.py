import numpy as np

from .audio import AudioWaveform
from .tokenizer import AdaptiveAudioAmplitudeTokenizer

def _monotonic_sin_waveform(tokenizer):
    sampling_rate = tokenizer.sampling_rate
    seconds_duration = 15
    waveform = np.sin(np.arange(0, 9, 10/(seconds_duration*sampling_rate)))

    wf_sr = AudioWaveform(waveform, sampling_rate)

    return wf_sr

def _silence_waveform(tokenizer):
    sampling_rate = tokenizer.sampling_rate
    seconds_duration = 15
    waveform = np.zeros([ seconds_duration*sampling_rate ])

    wf_sr = AudioWaveform(waveform, sampling_rate)

    return wf_sr


def test_pretokenize():
    tokenizer = AdaptiveAudioAmplitudeTokenizer()
    wf_sr = _silence_waveform(tokenizer)

    pretokonized_boarders = tokenizer.pretokenize(wf_sr.waveform)
    assert pretokonized_boarders == [ wf_sr.waveform.shape[-1] ]


def test_tokenizer():
    tokenizer = AdaptiveAudioAmplitudeTokenizer()
    wf_sr = _monotonic_sin_waveform(tokenizer)

    segments = tokenizer.tokenize(wf_sr)

    print("segments", len(segments))

    assert len(segments) == wf_sr.duration_seconds * 1000 // tokenizer.max_segment_duration_milliseconds

    sum_frames = sum(x.waveform.shape[-1] for x in segments)
    assert sum_frames == wf_sr.waveform.shape[-1]

    print("sum_frames", sum_frames, "waveform.shape[-1]", wf_sr.waveform.shape[-1])

