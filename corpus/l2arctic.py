import re
from tqdm import tqdm
from pathlib import Path
import os
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from builtins import str as unicode


def preprocess_text(text):
    text = unicode(text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Mistress")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text


class L2ArcticDataset(Dataset):
    """Dataset for L2Arctic corpus.
    
    Supports three modes:
    1. Single speaker: path = /path/to/L2Arctic/Arabic/ABA (has wav/ and transcript/ subdirs)
    2. Single accent: path = /path/to/L2Arctic/Arabic (contains multiple speaker directories)
    3. All accents: path = /path/to/L2Arctic (contains Arabic, Chinese, Hindi, etc.)
    
    Returns 4 values like librispeechc.py: (duration, audio_array, label, file_id)
    """
    
    # Known corrupted files to skip
    EXCEPTIONS = {
        "ABA": ["arctic_a0158", "arctic_b0013", "arctic_b0398"],
        "ASI": ["arctic_b0013"],
        "BWC": ["arctic_b0013", "arctic_b0345"],
        "EBVS": ["arctic_b0408"],
        "HJK": ["arctic_b0013"],
        "HKK": ["arctic_b0013"],
        "LXC": ["arctic_b0013"],
        "NCC": ["arctic_b0013"],
        "NJS": ["arctic_b0013"],
        "RRBI": ["arctic_a0298", "arctic_b0013"],
        "SKA": ["arctic_b0358"],
        "TNI": ["arctic_b0013"],
        "YBAA": ["arctic_a0094", "arctic_b0013"],
        "YDCK": ["arctic_b0013"],
        "YKWK": ["arctic_b0013"],
    }
    
    def __init__(self, bucket_size, path, enhance=False, ascending=False):
        """Initialize L2Arctic dataset.
        
        Args:
            bucket_size: Batch size (kept for API consistency)
            path: Path to dataset. Can be:
                  - Single speaker dir (e.g., /L2Arctic/Arabic/ABA)
                  - Accent dir (e.g., /L2Arctic/Arabic) 
                  - Root dir (e.g., /L2Arctic)
            enhance: Not used (kept for API consistency)
            ascending: Sort by text length ascending (default: descending)
        """
        self.path = path
        self.bucket_size = bucket_size
        
        # Detect mode and collect all speaker directories
        speaker_dirs = self._detect_and_collect_speakers(path)
        
        # Collect all files from all speakers
        file_list, text_list = [], []
        for speaker_dir in speaker_dirs:
            speaker_name = os.path.basename(speaker_dir)
            apath = os.path.join(speaker_dir, "wav")
            tpath = os.path.join(speaker_dir, "transcript")
            
            if not os.path.exists(apath) or not os.path.exists(tpath):
                continue
            
            # Get all wav files
            wav_files = sorted([f for f in os.listdir(apath) if f.endswith(".wav")])
            txt_files = sorted([f for f in os.listdir(tpath) if f.endswith(".txt")])
            
            # Match wav and txt files
            for wav_file, txt_file in zip(wav_files, txt_files):
                basename = wav_file.replace('.wav', '')
                
                # Skip corrupted files
                if speaker_name in self.EXCEPTIONS and basename in self.EXCEPTIONS[speaker_name]:
                    continue
                
                wav_path = os.path.join(apath, wav_file)
                txt_path = os.path.join(tpath, txt_file)
                
                with open(txt_path, "r") as f:
                    txt = f.read()
                    txt = preprocess_text(txt)
                
                file_list.append(wav_path)
                text_list.append(txt)
        
        # Sort by text length
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text_list), 
                                                                   reverse=not ascending, 
                                                                   key=lambda x:len(x[1]))])
        
        print(f'[INFO] L2Arctic dataset loaded from {path}')
        print(f'[INFO] Found {len(self.file_list)} samples from {len(speaker_dirs)} speaker(s)')

    def _detect_and_collect_speakers(self, path):
        """Detect the type of path and collect all speaker directories.
        
        Returns:
            List of speaker directory paths
        """
        # Check if this is a single speaker directory (has wav/ and transcript/)
        if os.path.exists(os.path.join(path, "wav")) and os.path.exists(os.path.join(path, "transcript")):
            # Mode 1: Single speaker
            return [path]
        
        # Check if subdirectories have wav/ and transcript/ (accent directory)
        speaker_dirs = []
        has_speaker_subdirs = False
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if not os.path.isdir(item_path):
                continue
                
            # Check if this is a speaker directory
            if os.path.exists(os.path.join(item_path, "wav")) and os.path.exists(os.path.join(item_path, "transcript")):
                speaker_dirs.append(item_path)
                has_speaker_subdirs = True
            # Check if this is an accent directory (contains speaker directories)
            else:
                for sub_item in os.listdir(item_path):
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        if os.path.exists(os.path.join(sub_item_path, "wav")) and os.path.exists(os.path.join(sub_item_path, "transcript")):
                            speaker_dirs.append(sub_item_path)
        
        if speaker_dirs:
            return sorted(speaker_dirs)
        
        raise ValueError(f"Could not detect valid L2Arctic structure at {path}")

    def __getitem__(self, index):
        """Return 4 values: (duration, audio_array, label, file_id)
        
        Format matches librispeechc.py for compatibility with baseline strategy.
        """
        file_path = self.file_list[index]
        label = self.text[index]
        
        # Load audio
        wav, sr = sf.read(file_path)
        wav = wav.astype(np.float32)
        
        # Resample if needed
        if sr != 16000:
            import torchaudio
            import torch
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)
            wav_tensor = torchaudio.transforms.Resample(sr, 16000)(wav_tensor)
            wav = wav_tensor.squeeze(0).numpy()
        
        # Calculate duration in seconds
        duration = len(wav) / 16000
        
        # Extract file id from path (include speaker name for uniqueness)
        # e.g., /path/Arabic/ABA/wav/arctic_a0001.wav -> ABA_arctic_a0001
        parts = file_path.split(os.sep)
        speaker_name = parts[-3]  # Get speaker directory name
        basename = os.path.basename(file_path).replace('.wav', '')
        file_id = f"{speaker_name}_{basename}"
        
        return duration, wav, label, file_id

    def __len__(self):
        return len(self.file_list)