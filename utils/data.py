import torch
torch.manual_seed(0)
import torchaudio
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset

SAMPLE_RATE = 16000

def multi_collate_fn(batch, processor):
    audio_lengths, noisy_audios, translations, file_names = zip(*batch)

    inputs = processor(noisy_audios[0], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    return audio_lengths, input_features, translations, file_names

def simple_collate_fn(batch):
    """Simple collate function that just returns the batch as-is for single samples."""
    # For batch_size=1, batch is a list with one tuple: [(len, audio, text, idx)]
    # Return the tuple directly
    if len(batch) == 1:
        return batch[0]
    # For batch_size > 1, return as list of tuples
    return batch

def collect_audio_batch(batch, extra_noise=0., maxLen=600000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        
        wav, sample_rate = torchaudio.load(filepath)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(wav)
        wav = wav.reshape(-1)
        if wav.shape[-1] >= maxLen:
            print(f'{filepath} has len {wav.shape}, truncate to {maxLen}')
            wav = wav[:maxLen]
            print(wav.shape)
        wav += extra_noise * torch.randn_like(wav)
        
        return wav

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            feat = audio_reader(str(b[0])).numpy()
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(b[1])

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_len, audio_feat, text, file


def create_dataset(split, name, path, batch_size=1, **kwargs):
    ''' Interface for creating all kinds of dataset'''
    loader_bs = batch_size

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    elif name.lower() =="l2arctic":
        from corpus.l2arctic import L2ArcticDataset as Dataset
        dataset = Dataset(batch_size, path)
        print(f'[INFO]    There are {len(dataset)} samples.')
        return dataset, loader_bs
    elif name.lower() == "noisylibri":
        from corpus.noisylibri import noisyLibriDataset as Dataset
    elif name.lower() == "multilibri":
        from corpus.multilibri import multiLingualLibriDataset as Dataset
        dataset = Dataset(split, batch_size, path, kwargs["noise_dir"], kwargs["snr"])
        print(f'[INFO]    There are {len(dataset)} samples.')
        return dataset, loader_bs
    elif name.lower() == "librispeechc":
        from corpus.librispeechc import LibrispeechCDataset as Dataset
        dataset = Dataset(split, batch_size, path, kwargs.get("lang"), kwargs.get("noise_dir"), kwargs.get("snr", 0), kwargs.get("task", "transcribe"))
        return dataset, loader_bs
    else:
        raise NotImplementedError

    dataset = Dataset(split, batch_size, path)
    print(f'[INFO]    There are {len(dataset)} samples.')

    return dataset, loader_bs


def load_SUTAdataset(split=None, name='librispeech', path=None, batch_size=1, extra_noise=0., num_workers=4, noise_dir=None, snr=0., lang="en", task='translation',is_multi_gpus=False, processor=None):
    ''' Prepare dataloader for training/validation'''
    dataset, loader_bs = create_dataset(split, name, path, batch_size, noise_dir=noise_dir, snr=snr, lang=lang, task=task)
    collate_fn = None
    if name.lower() in ['multilibri', 'librispeechc', 'l2arctic']:
        # These datasets return (audio_length, audio_array, label, index) directly
        # Use simple collate to avoid default stacking behavior
        collate_fn = simple_collate_fn
    elif name.lower() not in ['multilibri', 'covost2', 'fleurs', 'voxpopuli', 'tedlium', 'librispeechc']:
        collate_fn = partial(collect_audio_batch, extra_noise=extra_noise)

    # Create DistributedSampler
    if is_multi_gpus:
        sampler = DistributedSampler(dataset)
        my_collate_fn = partial(multi_collate_fn, processor=processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            sampler=sampler, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=my_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=loader_bs, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=num_workers, 
        )
         
    return dataloader
