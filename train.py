import torch
import os
from datetime import datetime
from tqdm import tqdm
from data_module import ReverbDataset
from torch.utils.data import DataLoader
from network import ResNet34Encoder, ConformerEncoder
from loss_fn import info_nce_loss

device="cuda" if torch.cuda.is_available() else "cpu"

# Logging and checkpointing setup
checkpoint_path = "TestBeforePaperSubmission"
log_file = checkpoint_path + "/training_log.txt"
os.makedirs(checkpoint_path, exist_ok=True)

# Create a timestamped header for new training runs
with open(log_file, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write(f"Training run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")
    f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Valid Loss':>12}\n")
    f.write("-"*40 + "\n")

# Data paths (Clean speech samples & RIRs directory)
dataset_path = "/data/khanagha/spgpu1_backup/VCTK_reverb/train/clean"
rirs_path = "/data/khanagha/spgpu1_backup/VCTK_reverb/train/rir"
valid_dataset_path = "/data/khanagha/spgpu1_backup/VCTK_reverb/valid/clean"
valid_rirs_path = "/data/khanagha/spgpu1_backup/VCTK_reverb/valid/rir"


# Loading the data
batch_size = 16
reverb_ds = ReverbDataset(dataset_path, rirs_path)
valid_reverb_ds = ReverbDataset(valid_dataset_path, valid_rirs_path)
train_data = DataLoader(reverb_ds, batch_size=batch_size, shuffle=True, num_workers=6)
valid_data = DataLoader(valid_reverb_ds, batch_size=batch_size, shuffle=False, num_workers=6)


# Choosing between resnet or conformer enc
encoder = ConformerEncoder().to(device)
# encoder = ResNet34Encoder(embedding_dim=256).to(device)

# optim = torch.optim.Adam(encoder.parameters(), lr = 1e-5)
optim = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-2)

epochs = 1
best_val_loss = float('inf')  
# Lambda for negative loss component
ld = 0.2

for ep in range(1, epochs + 1):
    encoder.train()
    train_loss = 0.0
    
    # Training loop
    train_pbar = tqdm(train_data, desc=f"Epoch [{ep}/{epochs}] Training", leave=False)
    for utt1_rirA, utt2_rirA, utt1_rirB, lab1, lab2, num_frames1, num_frames2, num_frames3 in train_pbar:
        # print(utt1_rirA.shape, utt1_rirA.transpose(1, 2).shape)

        z1 = encoder(utt1_rirA.transpose(1, 2).to(device), num_frames1.to(device))
        z2 = encoder(utt2_rirA.transpose(1, 2).to(device), num_frames2.to(device))
        z3 = encoder(utt1_rirB.transpose(1, 2).to(device), num_frames3.to(device))
        
        loss_pos = info_nce_loss(z1, z2, lab1.to(device))
        # loss_neg = 1 - (z1 * z3).sum(dim=-1).mean()
        loss_neg = (z1 * z3).sum(dim=-1).mean()
        loss = loss_pos + ld * loss_neg

        optim.zero_grad()
        loss.backward()

        optim.step()

        train_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_train_loss = train_loss / len(train_data)

    # Validation loop
    encoder.eval()
    valid_loss = 0.0
    valid_pbar = tqdm(valid_data, desc=f"Epoch [{ep}/{epochs}] Validation", leave=False)
    with torch.no_grad():
        for utt1_rirA, utt2_rirA, utt1_rirB, lab1, lab2, num_frames1, num_frames2, num_frames3 in valid_pbar:
            z1 = encoder(utt1_rirA.transpose(1, 2).to(device), num_frames1.to(device))
            z2 = encoder(utt2_rirA.transpose(1, 2).to(device), num_frames2.to(device))
            z3 = encoder(utt1_rirB.transpose(1, 2).to(device), num_frames3.to(device))

            loss_pos = info_nce_loss(z1, z2, lab1.to(device))
            # loss_neg = 1 - (z1 * z3).sum(dim=-1).mean()
            loss_neg = (z1 * z3).sum(dim=-1).mean()
            loss = loss_pos + ld * loss_neg

            valid_loss += loss.item()

            valid_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_valid_loss = valid_loss / len(valid_data)

    # Print per-epoch summary
    print(f"Epoch [{ep}/{epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Valid Loss: {avg_valid_loss:.4f}")

    with open(log_file, "a") as f:
        f.write(f"{ep:6d} | {avg_train_loss:12.6f} | {avg_valid_loss:12.6f}\n")

    # Checkpoint: Save if validation loss improves
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save({
            'epoch': ep,
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'valid_loss': best_val_loss
        }, checkpoint_path + f"/best_ckpt.pt")

# Final Epoch
torch.save({
    'epoch': ep,
    'encoder_state_dict': encoder.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'valid_loss': best_val_loss
}, checkpoint_path + f"/final_ckpt.pt")


