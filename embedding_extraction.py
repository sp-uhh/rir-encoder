from network import ConformerEncoder, ResNet34Encoder
import torch
import torchaudio
import os
from tqdm import tqdm

# feature extraction function
def emb_stft(wav):
    n_fft = 128
    hop_length = 64 
    window_length=128 
    window = torch.hann_window(window_length)
    wav_stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=window, return_complex=True)
    log_mag_stft = torch.log1p(torch.abs(wav_stft))
    return log_mag_stft

# Paths 
audio_paths = "<Path_to_audio_files>" 
output_path = "<Path_to_save_embeddings>"

os.makedirs(output_path, exist_ok=True)



# Room encoder config
conformer_encoder = ConformerEncoder()
checkpoint = torch.load('<Path_to_checkpoint>')
conformer_encoder.load_state_dict(checkpoint['encoder_state_dict'])
conformer_encoder.eval()


all_audio_paths = []
for root, dirs, files in os.walk(audio_paths):
        for f in files:
            if f.endswith('wav'):
                all_audio_paths.append(os.path.join(root,f))


for utt_path in tqdm(all_audio_paths):
     wave, sr = torchaudio.load(utt_path)
     log_mag_stft = emb_stft(wave)
     with torch.no_grad():
        x = conformer_encoder(log_mag_stft.transpose(1, 2), torch.tensor([log_mag_stft.size(-1)]))
        path_parts = utt_path.split("/")
        emb_path = os.path.join(output_path, path_parts[-1]).replace(".wav", ".pt")
        torch.save(x, emb_path)

    #  print(utt_path, path_parts, emb_path)

