import logging

import datasets
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from aat.training.config import TrainConfig, SegmentationType
from aat.tokenizer import AdaptiveAudioAmplitudeTokenizer
from aat.training.collate import TokenizedAudioWaveformCollator, NoSegmentationAudioWaveformCollator



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def build_collate_fn(train_config: TrainConfig, tokenizer, max_segment_waveform_frames=None, validation=False):
    
    if train_config.not_segmented_dataset:
        return NoSegmentationAudioWaveformCollator(train_config, tokenizer)
    
    max_segment_duration_milliseconds = int(max_segment_waveform_frames * 1000 / train_config.sampling_rate)
    audio_tokenizer = AdaptiveAudioAmplitudeTokenizer(
        max_segment_duration_milliseconds=max_segment_duration_milliseconds,
    )

    n_words = None if validation else train_config.n_words
    noise_augmentation = False if validation else True

    return TokenizedAudioWaveformCollator(
        train_config,
        audio_tokenizer,
        tokenizer,
        n_words=n_words,
        noise_augmentation=noise_augmentation,
        sampling_rate=train_config.sampling_rate,
        max_segment_waveform_frames=train_config.max_segment_waveform_frames,
    )

def build_train_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    shuffle = False
    if train_config.few_train_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_train_samples))
        shuffle = True

    persistent_workers = True if train_config.dataloader_num_workers > 0 else False

    return DataLoader(audio_stt_dataset, collate_fn=build_collate_fn(train_config, tokenizer),
                      batch_size=train_config.train_batch_size, num_workers=train_config.dataloader_num_workers,
                      drop_last=True, pin_memory=True, shuffle=shuffle, persistent_workers=persistent_workers)


def build_val_dataloader(audio_stt_dataset, train_config: TrainConfig, tokenizer):

    if train_config.few_val_samples is not None:
        audio_stt_dataset = audio_stt_dataset.select(range(train_config.few_val_samples))

    return DataLoader(audio_stt_dataset,
                      collate_fn=build_collate_fn(train_config, tokenizer, validation=True),
                      batch_size=train_config.val_batch_size, pin_memory=True)

def build_dataloaders(train_config: TrainConfig, tokenizer):

    # dataset_files = [ f'libris/train-{i:05}-of-00064.parquet' for i in range(train_config.dataset_shards) ] # 1 shard = 1 gb of data
    # logger.info(f"dataset_files {dataset_files}")
    # if train_config.few_train_samples:
    #     assert train_config.dataset_shards == 1, 'only one dataset shard is allowed with few_train_samples due to streaming is not possible with few samples'
    #     audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=False)
    # else:
    #     audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=dataset_files, streaming=True)
    #     audio_dataset = audio_dataset.shuffle(buffer_size=1000, seed=42)

    # test_dataset_files = [ f'libris/train-00063-of-00064.parquet' ] # 1 shard = 1 gb of data
    # audio_dataset_val = datasets.load_dataset("nguyenvulebinh/asr-alignment", split=datasets.Split.TRAIN, data_files=test_dataset_files, streaming=False)

    if train_config.not_segmented_dataset:
        audio_dataset = datasets.load_dataset("nguyenvulebinh/asr-alignment", 'libris')
        audio_dataset_val = audio_dataset['valid']
        audio_dataset =  audio_dataset['train']
        audio_dataset = audio_dataset.shuffle(seed=42)
    else:
        audio_dataset = datasets.load_from_disk(train_config.train_dataset_path)


        # audio_dataset = audio_dataset.to_iterable_dataset()
        audio_dataset = audio_dataset.shuffle(seed=42)
        # audio_dataset = audio_dataset.to_iterable_dataset()
        # audio_dataset = audio_dataset.shuffle(seed=42, buffer_size=1000)

        audio_dataset_val = datasets.load_from_disk(train_config.validation_dataset_path)

    # audio_dataset = load_dataset("nguyenvulebinh/asr-alignment", 'libris', split=datasets.Split.TRAIN, streaming=True)
    # audio_dataset.cast_column('audio', datasets.Audio(sampling_rate=train_config.sampling_rate))

    audio_dataset_val = audio_dataset_val.train_test_split(test_size=1000, seed=1)
    audio_dataset_val = audio_dataset_val['test']
    # TODO enshure val dataset is 16kHz sampling rate
    # audio_dataset_val.cast_column('audio', datasets.Audio(sampling_rate=train_config.sampling_rate))

    logger.info("load train dataloader")
    train_dataloader = build_train_dataloader(
        audio_dataset, train_config, tokenizer
    )
    logger.info("load val dataloader")
    val_dataloader = build_val_dataloader(
        audio_dataset_val, train_config, tokenizer
    )

    return train_dataloader, val_dataloader
