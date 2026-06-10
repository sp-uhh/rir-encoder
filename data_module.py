import numpy as np
import torchaudio
import pyroomacoustics as pra
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch
import random
# from pyroomacoustics.acoustics import measure_rt60, clarity
import glob
import os
import json
from pathlib import Path
import warnings
import pandas as pd
warnings.filterwarnings("ignore")





class ReverbDataset(Dataset):
    def __init__(self, clean_dir, rir_dir, sample_rate=16000, target_length_seconds=4):
        self.clean_files = sorted(list(Path(clean_dir).rglob("*.wav")))
        self.rirs = sorted(list(Path(rir_dir).rglob("*.wav")))
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.labels = {}
        self.labels_counter = 0

    def __len__(self):
        return len(self.clean_files)

    def _load_audio(self, path, target_sr):
        wave, sr = torchaudio.load(path)
        if sr != target_sr:
            wave = torchaudio.functional.resample(wave, sr, target_sr)
        return wave.squeeze(0)

    def _length_equalizer(self, wav, target_len, random_crop=True):
        target_num_samples = target_len * self.sample_rate
        true_len = wav.shape[-1]

        if true_len > target_num_samples:
            if random_crop:
                start = torch.randint(
                    0, true_len - target_num_samples + 1, (1,)
                ).item()
                wav = wav[start:start + target_num_samples]
            else:
                wav = wav[:target_num_samples]
            true_len = target_num_samples
        else:
            pad = target_num_samples - true_len
            wav = torch.nn.functional.pad(
                wav, (0, pad), mode="constant"
            )

        return wav, true_len

    def _stft(self, wav, orig_len, n_fft=128, hop_length=64, window_length=128):
        window = torch.hann_window(window_length)

        wav_stft = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=False,
            return_complex=True,
        )

        log_mag_stft = torch.log1p(torch.abs(wav_stft))

        # Number of frames that correspond to real (unpadded) audio
        if orig_len >= n_fft:
            valid_frames = 1 + (orig_len - n_fft) // hop_length
        else:
            valid_frames = 0

        valid_frames = min(valid_frames, wav_stft.shape[-1])

        return log_mag_stft, valid_frames

    def get_label(self, label):
        if label in self.labels:
            return self.labels[label]
        else:
            self.labels[label] = self.labels_counter
            self.labels_counter += 1
            return self.labels_counter - 1

    def __getitem__(self, idx):
        # Randomly sample two RIRs
        rir_path_a, rir_path_b = random.sample(self.rirs, 2)
        rir_a = self._load_audio(rir_path_a, self.sample_rate)
        rir_b = self._load_audio(rir_path_b, self.sample_rate)

        # Randomly sample two utterances
        clean_path_1, clean_path_2 = random.sample(self.clean_files, 2)
        utt1 = self._load_audio(clean_path_1, self.sample_rate)
        utt2 = self._load_audio(clean_path_2, self.sample_rate)

        # Convolution
        utt1_rirA = torch.tensor(
            pra.signal.fftconvolve(utt1.numpy(), rir_a.numpy(), mode="full"),
            dtype=torch.float32,
        )
        utt2_rirA = torch.tensor(
            pra.signal.fftconvolve(utt2.numpy(), rir_a.numpy(), mode="full"),
            dtype=torch.float32,
        )
        utt1_rirB = torch.tensor(
            pra.signal.fftconvolve(utt1.numpy(), rir_b.numpy(), mode="full"),
            dtype=torch.float32,
        )

        utt1_rirA, len1 = self._length_equalizer(utt1_rirA, self.target_length_seconds)
        utt2_rirA, len2 = self._length_equalizer(utt2_rirA, self.target_length_seconds)
        utt1_rirB, len3 = self._length_equalizer(utt1_rirB, self.target_length_seconds)

        utt1_rirA, num_frames1 = self._stft(utt1_rirA, orig_len=len1)
        utt2_rirA, num_frames2 = self._stft(utt2_rirA, orig_len=len2)
        utt1_rirB, num_frames3 = self._stft(utt1_rirB, orig_len=len3)

        return (
            utt1_rirA,
            utt2_rirA,
            utt1_rirB,
            self.get_label(rir_path_a.stem),
            self.get_label(rir_path_b.stem),
            num_frames1,
            num_frames2,
            num_frames3,
        )





class simpleDataset(Dataset):
    def __init__(self, data_dir, csv_dir, sample_rate=16000, target_length_seconds = 4):
        df = pd.read_csv(csv_dir)
        self.t60_dict = dict(zip(df["id"], df["t60"]))
        self.files = sorted(list(Path(data_dir).rglob("*.wav")))
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.labels = {}

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path, target_sr):
        wave, sr = torchaudio.load(path)
        if sr != target_sr:
            wave = torchaudio.functional.resample(wave, sr, target_sr)
        return wave.squeeze(0)

    def _length_equalizer(self, wav, target_len, random_crop=True):
        target_num_samples = target_len * self.sample_rate
        wav_length = wav.shape[-1]
        pad = max(0, target_num_samples - wav_length)
        
        # Pad or crop depending on length
        if pad == 0:
            if random_crop:
                rand_index = torch.randint(0,  wav_length - target_num_samples, size = (1,)).item() if wav_length - target_num_samples != 0 else 0
                wav = wav[..., rand_index:rand_index+target_num_samples]
            else:
                wav = wav[..., :target_num_samples]
        else:
            wav = torch.nn.functional.pad(wav, (pad//2, pad//2 + pad%2), mode = 'constant')
        return wav, wav_length
    
    def _stft(self, wav, n_fft = 128, hop_length = 64, window_length=128, orig_len=None):
        window = torch.hann_window(window_length)
        wav_stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=window, return_complex=True)
        log_mag_stft = torch.log1p(torch.abs(wav_stft))


        valid_frames = wav_stft.shape[-1]
        
        return log_mag_stft, valid_frames
    
    def get_label(self, label):
        if label in list(self.labels.keys()):
            return self.labels[label]
        else:
            self.labels[label] = self.labels_counter
            self.labels_counter += 1
            return self.labels_counter-1

    def __getitem__(self, idx):

        audio_id = self.files[idx].stem
        label = self.t60_dict[audio_id]

        file_path = self.files[idx]
        utt = self._load_audio(file_path, self.sample_rate)

        utt, len1 = self._length_equalizer(utt, self.target_length_seconds)

        # Spec transform
        utt, num_valid_frames_1 = self._stft(utt, orig_len=len1)



        return utt, label, num_valid_frames_1

class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, csv_dir, mode):
        df = pd.read_csv(csv_dir)

        if mode == "t60":
            label_col = "t60"
        elif mode == "c50":
            label_col = "c50"
        else:
            raise ValueError("Mode should be either 't60' or 'c50'")

        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df[np.isfinite(df[label_col])].reset_index(drop=True)

        self.label_dict = dict(zip(df["id"], df[label_col]))

        all_files = sorted(list(Path(data_dir).rglob("*.pt")))

        self.files = [
            f for f in all_files
            if f.stem in self.label_dict
        ]

        print(f"Found {len(all_files)} embedding files")
        print(f"Using {len(self.files)} files with valid {mode} labels")
        print(f"Skipped {len(all_files) - len(self.files)} files")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        room_emb = torch.load(self.files[idx])
        audio_id = self.files[idx].stem
        label = self.label_dict[audio_id]
        return room_emb, label