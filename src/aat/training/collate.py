from typing import Callable, List, Dict

import numpy as np
import torch
import os

from transformers import AutoProcessor

from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.audio import AudioWaveform
from aat.training.config import TrainConfig, SegmentationType

import random

from transformers.utils import PaddingStrategy

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function


PREFIXES = [
    "The audio transcription states:",
    "According to the audio transcript:",
    "As per the audio transcription:",
    "In the audio recording it is said:",
    "Based on the audio script:",
    "Per the audio record:",
    "From the audio file it can be heard:",
    "What the audio text conveys is:",
    "Transcribed from the audio:",
    "Listening to the recording reveals:",
]


class PadWaveformsMixin():
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


class TokenizedAudioWaveformCollator(PadWaveformsMixin):

    def __init__(self,
                 segmentation,
                 train_config: TrainConfig,
                 audio_tokenizer: AdaptiveAudioAmplitudeTokenizer,
                 tokenizer,
                 n_words=None,
                 noise_augmentation: bool = False,
                 uniform_segmentation_frames_per_segment=None,
                ):
        self.train_config = train_config
        
        self.segmentation = segmentation
        self.uniform_segmentation_frames_per_segment = uniform_segmentation_frames_per_segment

        assert self.segmentation != SegmentationType.none

        self.n_words = n_words

        self.max_segment_waveform_frames = audio_tokenizer.max_segment_frames
        self.sampling_rate = audio_tokenizer.sampling_rate

        self.noise_augmentation = noise_augmentation

        self.audio_tokenizer = audio_tokenizer
        self.tokenizer = tokenizer

        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        
        self.melspec_base_path = "data/libris_melspectrograms"
        self.melspec_files = set(os.listdir(self.melspec_base_path))
        
        return

    def __call__(self, items):

        tokenizer = self.tokenizer

        result = dict()
        # random select caption
        bos_token = tokenizer.decode(tokenizer.bos_token_id)
        eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer_input = []
        tokenizer_input_prefixes_for_validation = []

        segments_boarders = []
        audio_segments_waveforms = []
        segments_max_frame_len = []

        n_words = None
        if self.n_words is not None:
            n_words = random.randint(5, self.n_words)

        items_melspecs = []
        for i, item in enumerate(items):
            waveform = np.array(item['audio']['array'])
            sampling_rate = item['audio']['sampling_rate']
            assert sampling_rate == self.sampling_rate

            if self.noise_augmentation:
                waveform += np.random.rand(waveform.shape[-1]) * random.randint(1, 50) / 1000

            waveform_num_frames = waveform.shape[-1]

            if self.segmentation == SegmentationType.uniform:
                num_segments = waveform_num_frames // self.uniform_segmentation_frames_per_segment
                segments_list = [ self.uniform_segmentation_frames_per_segment ] * num_segments

                if waveform_num_frames % self.uniform_segmentation_frames_per_segment > 0:
                    segments_list.append(waveform_num_frames -  sum(segments_list))

                frames_boarders_raw = np.array(segments_list)
                frames_boarders = frames_boarders_raw.cumsum()
            elif self.segmentation == SegmentationType.adaptive:
                waveform_normed = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
                awf_sr = AudioWaveform(waveform_normed, sampling_rate)
                
                melspec = None
                melspec_file_path = os.path.join(self.melspec_base_path, item['id'])
                if item['id'] in self.melspec_files:
                    try:
                        melspec = torch.load(melspec_file_path, weights_only=False)
                    except Exception as e:
                        print(f"Failed to load {melspec_file_path}:", e)

                item_audio_segments, melspec = self.audio_tokenizer.tokenize(awf_sr, melspec=melspec)
                items_melspecs.append(melspec)
                segment_frames = [ sf.waveform.shape[-1] for sf in item_audio_segments ]
                frames_boarders_raw = np.array(segment_frames)
                frames_boarders = frames_boarders_raw.cumsum()
            else:
                raise ValueError(f"Unhandled seglent projection type: {self.segmentation}")

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
                waveform_end_frame   = int(item['word_end'][word_end_idx-1] * self.sampling_rate)

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
            prefix_for_validation = ""
            if self.train_config.add_prefix:
                prefix_for_validation = random.choice(PREFIXES) + " "
                item_text = prefix_for_validation + item_text

            prefix_for_validation = bos_token + prefix_for_validation
            text_for_item = bos_token + item_text + eos_token


            tokenizer_input.append(text_for_item)
            tokenizer_input_prefixes_for_validation.append(prefix_for_validation)

            audio_segments_waveforms.append(waveform)

            segments_boarders.append( frames_boarders )
            segments_max_frame_len.append(frames_boarders_raw.max())


        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])
        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])

        result['input_ids_attention_mask'] = result['attention_mask']

        tokenized_caption_prefix = tokenizer(tokenizer_input_prefixes_for_validation, padding=True)
        result['prefix_input_ids'] = torch.tensor(tokenized_caption_prefix['input_ids'])
        result['prefix_attention_mask'] = torch.tensor(tokenized_caption_prefix['attention_mask'])


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

        audio_processed_output = self.audio_processor(audio_segments_waveforms, padding=True, return_tensors="pt", sampling_rate=self.train_config.sampling_rate)

        audio_input_values = audio_processed_output.input_values
        audio_attention_mask = audio_processed_output.attention_mask
        
        assert audio_input_values.shape[1] > 0
        assert audio_attention_mask.shape[1] > 0

        max_melspec_items = int(1 + np.floor(self.max_segment_waveform_frames / self.audio_tokenizer.hop_length))
        batched_segments_melspectrograms = torch.zeros([batch_size, segments_count, self.audio_tokenizer.num_mel_filters, max_melspec_items])
        batched_segments = torch.zeros([batch_size, segments_count, max_segment_waveform_frames])
        segments_waveforms_mask = torch.zeros([batch_size, segments_count, max_segment_waveform_frames])
        assert len(items_melspecs) == batch_size
        for batch_i in range(batch_size):
            prev_segment_boarder = 0
            full_audio_log_mel_spectrogram = items_melspecs[batch_i]
            
            for segment_i in range(segments_count):
                segment_boarder = segments_boarders_padded[batch_i, segment_i]
                if segment_i > 0 and segment_boarder == 0:
                    continue

                assert prev_segment_boarder < segment_boarder
                segment_length = segment_boarder - prev_segment_boarder
                current_waveform = audio_input_values[batch_i, prev_segment_boarder:segment_boarder]
                batched_segments[batch_i, segment_i, :segment_length] = current_waveform
                segments_waveforms_mask[batch_i, segment_i, :segment_length] = 1

                melspec_boarder_start, melspec_boarder_end = ((prev_segment_boarder // self.audio_tokenizer.hop_length), (segment_boarder // self.audio_tokenizer.hop_length))
                # [ self.audio_tokenizer.num_mel_filters, melspec_seq_len ]
                segment_log_mel_spectrogram = full_audio_log_mel_spectrogram[:, melspec_boarder_start:melspec_boarder_end]
                batched_segments_melspectrograms[batch_i, segment_i, :, :segment_log_mel_spectrogram.shape[1]] = torch.from_numpy(segment_log_mel_spectrogram)

                prev_segment_boarder = segment_boarder

        result['batched_segments'] = batched_segments
        result['batched_segments_melspectrograms'] = batched_segments_melspectrograms
        result['segments_waveforms_mask'] = segments_waveforms_mask
        result['segments_count'] = segments_count

        # assert not result['batched_segments'].isnan().any()
        assert not result['segments_waveforms_mask'].isnan().any()

        return result



class NoSegmentationAudioWaveformCollator(PadWaveformsMixin):

    def __init__(self, train_config: TrainConfig, tokenizer):

        self.train_config = train_config
        self.sampling_rate = train_config.sampling_rate

        self.tokenizer = tokenizer

        self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

        return

    def __call__(self, items):

        tokenizer = self.tokenizer

        result = dict()
        # random select caption
        bos_token = tokenizer.decode(tokenizer.bos_token_id)
        eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer_input = []
        audio_waveforms = []
        tokenizer_input_prefixes_for_validation = []

        for i, item in enumerate(items):
            waveform = np.array(item['audio']['array'])
            # noise augmentation
            waveform += np.random.rand(waveform.shape[-1]) * random.randint(1, 50) / 1000

            words = item['words']
            prefix_for_validation = ""

            item_text = " ".join(words)
            prefix_for_validation = ""
            if self.train_config.add_prefix:
                prefix_for_validation = random.choice(PREFIXES) + " "
                item_text = prefix_for_validation + item_text

            prefix_for_validation = bos_token + prefix_for_validation
            text_for_item = bos_token + item_text + eos_token

            tokenizer_input.append(text_for_item)
            tokenizer_input_prefixes_for_validation.append(prefix_for_validation)

            audio_waveforms.append(waveform)

        tokenized_caption = tokenizer(tokenizer_input, padding=True)
        result['input_ids'] = torch.tensor(tokenized_caption['input_ids'])
        result['attention_mask'] = torch.tensor(tokenized_caption['attention_mask'])

        result['input_ids_attention_mask'] = result['attention_mask']

        tokenized_caption_prefix = tokenizer(tokenizer_input_prefixes_for_validation, padding=True)
        result['prefix_input_ids'] = torch.tensor(tokenized_caption_prefix['input_ids'])
        result['prefix_attention_mask'] = torch.tensor(tokenized_caption_prefix['attention_mask'])


        segments_padded_normalized_with_mask = self.audio_processor(audio_waveforms, return_tensors="pt", sampling_rate=self.train_config.sampling_rate, padding=PaddingStrategy.LONGEST)

        result['waveforms'] = segments_padded_normalized_with_mask.input_values
        result['waveforms_attention_mask'] = segments_padded_normalized_with_mask.attention_mask

        assert not result['waveforms'].isnan().any()
        assert not result['waveforms_attention_mask'].isnan().any()

        return result


