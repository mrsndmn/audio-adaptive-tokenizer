from typing import Callable, List, Dict

import numpy as np
import torch

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform
from aat.training.config import TrainConfig, SegmentationType

import random

class TokenizedAudioWaveformCollator():

    def __init__(self, train_config: TrainConfig, audio_tokenizer: AdaptiveAudioAmplitudeTokenizer, build_text_tokenizer: Callable, sampling_rate: int, max_segment_waveform_frames: int, n_words=None, noise_augmentation: bool = True):


        self.train_config = train_config
        self.sampling_rate = sampling_rate

        self.n_words = n_words

        self.max_segment_waveform_frames = max_segment_waveform_frames

        self.noise_augmentation = noise_augmentation

        self.audio_tokenizer = audio_tokenizer
        self.build_text_tokenizer = build_text_tokenizer
        self.tokenizer = self.build_text_tokenizer()

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

        n_words = None
        if self.n_words is not None:
            n_words = random.randint(5, self.n_words)

        for i, item in enumerate(items):
            waveform = np.array(item['audio']['array'])

            if self.noise_augmentation:
                waveform += np.random.rand(waveform.shape[-1]) * random.randint(1, 50) / 1000

            waveform_offset = 0
            if self.train_config.segment_boarders_noize:
                assert self.train_config.segmentation == SegmentationType.uniform, 'uniform segmentaion'
                waveform_offset = random.randint(0, int(self.train_config.uniform_segmentation_frames_per_segment // 2))
                waveform = waveform[waveform_offset:]

            waveform_num_frames = waveform.shape[-1]

            if self.train_config.segmentation == SegmentationType.uniform:
                num_segments = waveform_num_frames // self.train_config.uniform_segmentation_frames_per_segment
                segments_list = [ self.train_config.uniform_segmentation_frames_per_segment ] * num_segments
                segments_list.append(waveform_num_frames -  sum(segments_list))
                frames_boarders_raw = np.array(segments_list)
                frames_boarders = frames_boarders_raw.cumsum()
            elif self.train_config.segmentation == SegmentationType.uniform:
                frames_boarders_raw = np.array(item['segment_frames'])
                frames_boarders = frames_boarders_raw.cumsum()
            else:
                raise ValueError(f"Unhandled seglent projection type: {self.train_config.segmentation}")

            assert frames_boarders_raw.sum() == waveform_num_frames

            words = item['words']

            word_start_idx = 0
            word_end_idx = len(words) - 1
            waveform_start_frame = 0
            waveform_end_frame = waveform.shape[-1]

            if n_words is not None and len(words) > n_words:
                word_start_idx = random.randint(0, len(words)-n_words)
                word_end_idx = word_start_idx + n_words
                words = words[word_start_idx:word_end_idx]

                waveform_start_frame = int(item['word_start'][word_start_idx] * self.sampling_rate)
                waveform_end_frame   = int(item['word_end'][word_end_idx-1] * self.sampling_rate) - waveform_offset

                frames_boarders_with_zero = np.insert(frames_boarders, 0, [ 0 ])

                waveform_start_segment_idx = np.searchsorted(frames_boarders_with_zero, waveform_start_frame)
                waveform_start_segment_idx -= 1
                waveform_start_segment_idx = max(waveform_start_segment_idx, 0)
                assert waveform_start_segment_idx >= 0

                waveform_end_segment_idx = np.searchsorted(frames_boarders_with_zero, waveform_end_frame, side='right')
                assert waveform_end_segment_idx < len(frames_boarders_with_zero)

                start_segment_waveform_num = frames_boarders_with_zero[waveform_start_segment_idx]
                assert start_segment_waveform_num <= waveform_start_frame

                end_segment_waveform_num = frames_boarders_with_zero[waveform_end_segment_idx]
                assert end_segment_waveform_num >= waveform_end_frame

                frames_boarders = frames_boarders_with_zero[waveform_start_segment_idx:(waveform_end_segment_idx+1)]
                frames_boarders = frames_boarders - start_segment_waveform_num
                assert frames_boarders[0] == 0
                frames_boarders = frames_boarders[1:] # сut off leading zero
                assert len(frames_boarders) > 1

                waveform = waveform[start_segment_waveform_num:end_segment_waveform_num]

            item_text = " ".join(words)
            text_for_item = bos_token + item_text + eos_token


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

