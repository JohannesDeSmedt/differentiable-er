
import torch
from tqdm import tqdm
import math

import torch.nn as nn
from preprocessing import EventDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from model_help import EventTransformer, SDFAProjector, entropic_relevance_diff_loss


def collate_batch(batch):
    batch = [b.tolist() if isinstance(b, torch.Tensor) else b for b in batch]
    max_len = max(len(seq) for seq in batch)
    
    padded = []
    mask = []
    for seq in batch:
        padded_seq = seq + [0] * (max_len - len(seq))
        padded.append(padded_seq)
        mask.append([1] * len(seq) + [0] * (max_len - len(seq)))

    x = torch.tensor(padded, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    return x, mask, [torch.tensor(seq, dtype=torch.long) for seq in batch]


class SDFA_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.projector = SDFAProjector(d_model, sdfa_shape)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        soft_tensor = self.projector(encoded)
        return soft_tensor


def train_model(model, le, daily_sequences, optimizer, max_len, device, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        for day_idx, (sequences, target_sequences, target_dfg_tensor) in enumerate(daily_sequences):  # list of (X, y)
            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)

            batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Day {day_idx+1} Progress", leave=False)
            

            for x, mask, _ in batch_tqdm:  # sequences not needed here
                x, mask = x.to(device), mask.to(device)

                # Repeat target for each item in batch
                optimizer.zero_grad()
                sdfa_pred = model(x, mask)

                sdfa_target = target_dfg_tensor.to(device).unsqueeze(0).expand(sdfa_pred.shape[0], -1, -1)

                loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(daily_sequences):.4f}")

def evaluate_model(model, le, daily_sequences, max_len, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for day_idx, (sequences, target_sequences, target_dfg_tensor) in enumerate(daily_sequences):  # list of (X, y)
            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)

            test_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_batch)

            for x, mask, sequences in dataloader:
                x, mask = x.to(device), mask.to(device)

                sdfa_pred = model(x, mask)
                sdfa_target = target_dfg_tensor.to(device).unsqueeze(0).expand(sdfa_pred.shape[0], -1, -1)

                loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
                total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")