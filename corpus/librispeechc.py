import json
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset


class LibriSpeechCCorpus(object):
    """Corpus loader for LibriSpeech-C data."""
    
    def __init__(self, root: str) -> None:
        self.root = root
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav_path = f"{self.root}/wav/{basename}.wav"
        wav, sr = sf.read(wav_path)
        # Convert to float32 if needed
        wav = wav.astype(np.float32)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class LibrispeechCDataset(Dataset):
    """Dataset wrapper for LibriSpeech-C (corrupted LibriSpeech) corpus.
    
    Returns 4 values like fleurs.py: (duration, audio_array, label, id)
    The audio is already corrupted, so no additional noise is added.
    """
    
    def __init__(self, split, bucket_size, path, lang=None, noise_dir=None, snr=0., task='transcribe'):
        """
        Args:
            split: Not used for LibriSpeech-C (kept for API consistency)
            bucket_size: Batch size for bucketing
            path: Root path to LibriSpeech-C data (e.g., /path/to/LibriSpeech-c/GS/snr=10)
            lang: Language (kept for API consistency)
            noise_dir: Not used (kept for API consistency)
            snr: Not used (kept for API consistency)
            task: Task type (kept for API consistency)
        """
        self.bucket_size = bucket_size
        self.corpus = LibriSpeechCCorpus(root=path)
        
    def __getitem__(self, index):
        """Return 4 values: (duration, audio_array, label, id)
        
        Audio is already corrupted in LibriSpeech-C, no additional noise needed.
        Format matches fleurs.py for compatibility with baseline strategy.
        """
        item = self.corpus.get(index)
        audio_array = item['wav']  # numpy float32 array
        label = item['text']
        sample_id = item['id']
        duration = len(audio_array) / 16000
        
        # Return in same format as fleurs: (duration, audio_array, label, id)
        return duration, audio_array, label, sample_id

    def __len__(self):
        return len(self.corpus)
