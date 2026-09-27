import os
from pathlib import Path
import numpy as np
import torch
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm
import re
from builtins import str as unicode
from scipy.io import wavfile
import pickle

from . import Define


class LibriSpeechCorpus(object):
    def __init__(self, split=None) -> None:
        if split is None:
            self.root = "_cache/LibriSpeech"
        else:
            self.root = f"_cache/LibriSpeech/{split}"
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
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class LibriSpeechCCorpus(object):
    def __init__(self, root: str) -> None:
        self.root = root
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        # tt = 0
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)
                # tt += query["length"]
        # print(tt / 16000 / 60)  # about 5hr

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }

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


class StandardCorpus(object):
    def __init__(self, root: str) -> None:
        assert os.path.exists(root), f"Can't find {root}, please run preprocess first!"
        with open(f"{root}/data_info.json", "r", encoding="utf-8") as f:
            data_info = json.load(f)
        self.wav_paths = []
        self.texts = []
        
        for instance in data_info:
            basename = instance["basename"]
            with open(f"{root}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{root}/wav/{basename}.wav")

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> dict:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = preprocess_text(self.texts[idx])

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }

class L2ArcticCorpus(object):

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

    def __init__(self) -> None:
        self.root = Define.L2ARCTIC
        self._init_info()

    def _init_info(self):
        self.accents = []
        self.accent2spks = {}
        self.spk2accent = {}
        self.spks = []
        for accent in os.listdir(self.root):
            if not os.path.isdir(f"{self.root}/{accent}"):
                continue
            self.accents.append(accent)
            self.accent2spks[accent] = []
            for spk in os.listdir(f"{self.root}/{accent}"):
                self.spks.append(spk)
                self.accent2spks[accent].append(spk)
                self.spk2accent[spk] = accent
        assert len(self.spks) == 24

        self.spk2files = {}
        for spk in self.spks:
            files = []
            accent = self.spk2accent[spk]
            spk_root = f"{self.root}/{accent}/{spk}"
            for basename in os.listdir(f"{spk_root}/transcript"):
                basename = basename[:-4]
                if spk in L2ArcticCorpus.EXCEPTIONS and basename in L2ArcticCorpus.EXCEPTIONS[spk]:
                    continue
                files.append({
                    "wav_path": f"{spk_root}/wav/{basename}.wav",
                    "text_path": f"{spk_root}/transcript/{basename}.txt"
                })
            self.spk2files[spk] = files

    def get(self, spk: str, idx: int) -> dict:
        q = self.spk2files[spk][idx]
        wav, _ = librosa.load(q["wav_path"], sr=16000)
        with open(q["text_path"], 'r') as f:
            text = f.read()
        text = preprocess_text(text)

        return {
            "id": q["wav_path"],
            "wav": wav,
            "text": text
        }


if __name__ == "__main__":
    corpus = LibriSpeechCCorpus(root=f"_cache/LibriSpeech-c/GS/snr=5")
