from typing import Callable, List, Dict

import numpy as np
import torch

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform

import random

class TokenizedAudioWaveformCollator():

    def __init__(self, audio_tokenizer: AdaptiveAudioAmplitudeTokenizer, build_text_tokenizer: Callable, sampling_rate: int, max_segment_waveform_frames: int, validation: bool = False, augmentation: bool = True):

        self.sampling_rate = sampling_rate

        self.max_segment_waveform_frames = max_segment_waveform_frames

        self.augmentation = augmentation

        self.audio_tokenizer = audio_tokenizer
        self.build_text_tokenizer = build_text_tokenizer
        self.tokenizer = self.build_text_tokenizer()

        self.validation = validation

        return

    def pad_waveforms(self, waveforms_padding_list: List) -> Dict:
        assert len(waveforms_padding_list[0].shape) == 1, 'channel dim is not supported for waveform'
        max_len = max(x.shape[-1] for x in waveforms_padding_list)
        batch_size = len(waveforms_padding_list)

        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        batched_waveform = torch.zeros(batch_size, max_len)

        for i, wf in enumerate(waveforms_padding_list):
            attention_mask[i, :wf.shape[-1]] = 1
            batched_waveform[i, :wf.shape[-1]] = torch.from_numpy(wf)

        return {
            "input_values": batched_waveform,
            "attention_mask": attention_mask,
        }


    def __call__(self, items):

        tokenizer = self.tokenizer

        result = dict()
        # random select caption
        bos_token = tokenizer.decode(tokenizer.bos_token_id)
        eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer_input = []

        segments_boarders = []
        audio_segments_waveforms = []
        segments_max_frame_len = []

        for i, item in enumerate(items):
            words = item['words']
            first_word_second = 0
            last_word_second = item['word_end'][-1]

            item_text = " ".join(words)
            text_for_item = bos_token + item_text + eos_token

            frame_boarder_left = int(first_word_second * self.sampling_rate)
            frame_boarder_right = int(last_word_second * self.sampling_rate)

            waveform = np.array(item['audio']['array'])

            if self.augmentation:
                waveform += np.random.rand(waveform.shape[-1]) * 0.0003

            frames_boarders_raw = np.array(item['segment_frames'])
            frames_boarders = frames_boarders_raw.cumsum()

            tokenizer_input.append(text_for_item)

            audio_segments_waveforms.append(waveform)

            segments_boarders.append( frames_boarders )
            segments_max_frame_len.append(frames_boarders_raw.max())


        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])
        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])

        result['input_ids_attention_mask'] = result['attention_mask']

        audio_preprocessed = self.pad_waveforms(
            audio_segments_waveforms,
        )

        result['audio_input_values'] = audio_preprocessed['input_values']
        result['audio_attention_mask'] = audio_preprocessed['attention_mask']
        assert result['audio_input_values'].shape[1] > 0
        assert result['audio_attention_mask'].shape[1] > 0

        max_len_segments_boarders = max(len(x) for x in segments_boarders)
        segments_boarders_padded = torch.zeros([ result['attention_mask'].shape[0], max_len_segments_boarders ], dtype=torch.long)
        segments_boarders_attention_mask = torch.zeros_like(segments_boarders_padded)

        # todo vectorize
        for i, sb in enumerate(segments_boarders):
            segments_boarders_padded[i, :len(sb)] = torch.tensor(sb, dtype=torch.long)
            segments_boarders_attention_mask[i, :len(sb)] = 1

        result['segments_boarders_padded'] = segments_boarders_padded
        result['segments_boarders_attention_mask'] = segments_boarders_attention_mask
        result['segments_max_frame_len'] = torch.tensor(segments_max_frame_len)

        # make waveforms segments
        batch_size = segments_boarders_padded.shape[0]
        segments_count = segments_boarders_padded.shape[1]

        max_segment_waveform_frames = self.max_segment_waveform_frames
        batched_segments = torch.zeros([batch_size, segments_count, max_segment_waveform_frames])

        segments_waveforms_mask = torch.zeros_like(batched_segments)

        for batch_i in range(batch_size):
            prev_segment_boarder = 0
            for segment_i in range(segments_count):
                segment_boarder = segments_boarders_padded[batch_i, segment_i]
                if segment_i > 0 and segment_boarder == 0:
                    break
                segment_waveform = result['audio_input_values'][batch_i, prev_segment_boarder:segment_boarder]
                batched_segments[batch_i, segment_i, :segment_waveform.shape[0]] = segment_waveform
                segments_waveforms_mask[batch_i, segment_i, :segment_waveform.shape[0]] = 1
                prev_segment_boarder = segment_boarder

        result['batched_segments'] = batched_segments
        result['segments_waveforms_mask'] = segments_waveforms_mask

        return result

