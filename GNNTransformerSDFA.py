# STEP 0: MPS DEVICE SETUP
import torch
from tqdm import tqdm

# STEP 1: Transformer Encoder
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader


device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        emb = self.embedding(x)  # (B, T, D)
        emb = emb * mask.unsqueeze(-1)  # Apply mask
        encoded = self.encoder(emb)  # (B, T, D)
        return encoded

class GNNDecoder(nn.Module):
    def __init__(self, d_model, num_states):
        super().__init__()
        self.num_states = num_states
        self.fc_q = nn.Linear(d_model, num_states)
        self.fc_k = nn.Linear(d_model, num_states)

    def forward(self, encoded_seq):
        B, T, D = encoded_seq.shape

        q = self.fc_q(encoded_seq)  # (B, T, Q)
        k = self.fc_k(encoded_seq)  # (B, T, Q)

        q = q.unsqueeze(3)  # (B, T, Q, 1)
        k = k.unsqueeze(2)  # (B, T, 1, Q)

        logits = torch.sigmoid(q * k)  # outer product → (B, T, Q, Q)
        return logits

  
class TransformerGNNSDFA(nn.Module):
    def __init__(self, vocab_size, num_states, d_model=128, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # GNN Decoder: Maps transformer output to state transitions
        self.decoder = GNNDecoder(d_model, num_states)

    def forward(self, x, mask):
        emb = self.embedding(x)  # (B, T, D)
        emb = emb.transpose(0, 1)  # Transformer expects (T, B, D)
        encoded = self.transformer(emb, src_key_padding_mask=mask)  # (T, B, D)
        encoded = encoded.transpose(0, 1)  # back to (B, T, D)
        return self.decoder(encoded)  # (B, T, Q, Q)



def sdfa_loss(pred, target, eps=1e-9):
    pred = pred.clamp(min=eps, max=1 - eps)
    return -torch.mean(target * torch.log2(pred) + (1 - target) * torch.log2(1 - pred))

def train_model(model, le, daily_sequences, optimizer, max_len, num_epochs=10):
    model = model.to(device)
    model.train()
    
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0
        for date in sorted(daily_sequences.keys()):
            # print(f"Processing date: {date}")
            sequences = daily_sequences[date]
            # print(sequences[:5])  # Print first 5 sequences for verification
            # print('Size of sequences:', len(sequences))

            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)
                   
            batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Batch Progress", leave=False)
            for x, mask, sequences in batch_tqdm:   
                    # print(f"Batch size: {x.shape[0]}, Sequence length: {x.shape[1]}")
                    # print('x:', x)
                    # print('mask:', mask)
                    # print(f"Sequences: {sequences}")
                    x, mask = x.to(device), mask.to(device)

                    batch_targets = [
                        sequence_to_dfg_sdfa_tensor(seq.numpy(), le, num_states=num_states)
                        for seq in sequences
                    ]
                    sdfa_target = torch.stack(batch_targets).unsqueeze(1).repeat(1, max_len, 1, 1).to(device)

                    optimizer.zero_grad()
                    sdfa_pred = model(x, mask)
                    # print(f"Shape of sdfa_pred: {sdfa_pred.shape}, sdfa_target: {sdfa_target.shape}")
                    loss = sdfa_loss(sdfa_pred, sdfa_target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, le, daily_sequences, max_len):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():

        for date in sorted(daily_sequences.keys()):
            # print(f"Processing date: {date}")
            sequences = daily_sequences[date]
            # print(sequences[:5])  # Print first 5 sequences for verification

            test_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_batch)

            for x, mask, sequences in dataloader:
                x, mask = x.to(device), mask.to(device)

                batch_targets = [
                    sequence_to_dfg_sdfa_tensor(seq.numpy(), le, max_len=max_len)
                    for seq in sequences
                ]
                sdfa_target = torch.stack(batch_targets).to(device)
                sdfa_pred = model(x, mask)

                dfg = sdfa_to_dfg(sdfa_pred, le)
                print(f"DFG for date {date}: {dfg}")
                

                loss = sdfa_loss(sdfa_pred, sdfa_target)
                total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")


def sequence_to_dfg_sdfa_tensor(sequence, le, max_len):
    sigma_size = len(le.classes_)
    tensor = np.zeros((sigma_size, max_len, max_len), dtype=np.float32)

    labels = le.inverse_transform(sequence)
    for i in range(len(labels) - 1):
        a1 = labels[i]
        a2 = labels[i + 1]
        if a1 in le.classes_ and a2 in le.classes_:
            a_idx = le.transform([a1])[0]
            tensor[a_idx, i, i+1] += 1.0

    # Normalize each slice per activity
    for a in range(sigma_size):
        max_val = tensor[a].max()
        if max_val > 0:
            tensor[a] /= max_val

    return torch.tensor(tensor)


from torch.utils.data import Dataset

class EventDataset(Dataset):
    def __init__(self, sequences, pad_token=0):
        self.sequences = sequences
        self.pad_token = pad_token

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long)

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


def encode_activities(df, le):
    # Unknown activities in test get label -1
    known = set(le.classes_)
    df = df.copy()
    df["activity_encoded"] = df["activity"].apply(lambda x: le.transform([x])[0] if x in known else -1)
    return df


def extract_label_prefixes(df, prefix_len, le):
    # Ensure timestamp is datetime
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df['date'] = df['time:timestamp'].dt.date  # extract date part

    # Group by case_id
    cases = df.groupby("case_id")
    daily_prefixes = {}

    for _, case in cases:
        activities = case["activity"].tolist()
        timestamps = case["time:timestamp"].tolist()

        for i in range(1, min(len(activities), prefix_len) + 1):
            prefix = activities[:i]
            current_event_date = timestamps[i - 1].date()

            # Encode
            encoded_prefix = le.transform(prefix).tolist()

            if current_event_date not in daily_prefixes:
                daily_prefixes[current_event_date] = []

            daily_prefixes[current_event_date].append(encoded_prefix)

    return daily_prefixes

def sdfa_to_dfg(sdfa_tensor, le, threshold=0.5):
    dfg = {}
    sigma = le.classes_

    sdfa_collapsed = sdfa_tensor.sum(dim=(0, 1)) 
    print("SDFA tensor shape:", sdfa_tensor.shape)  # Expecting (B, T, V, V)
    print("Collapsed shape:", sdfa_collapsed.shape) #

    for i in range(sdfa_collapsed.shape[0]):
        for j in range(sdfa_collapsed.shape[1]):
            weight = sdfa_collapsed[i, j].item()
            if weight >= threshold:
                from_act = sigma[i]
                to_act = sigma[j]
                dfg_key = (from_act, to_act)
                dfg[dfg_key] = weight

    return dfg


