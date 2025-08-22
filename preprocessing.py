import torch
from tqdm import tqdm
import math

import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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
    # indien niet in test set -> label -1
    known = set(le.classes_)
    df = df.copy()
    df["activity_encoded"] = df["activity"].apply(lambda x: le.transform([x])[0] if x in known else -1)
    return df


def extract_daily_prefixes_with_shifted_targets(df, prefix_len, le, max_len=None):
    from collections import defaultdict

    # zie skpm
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df['date'] = df['time:timestamp'].dt.date

    cases = df.groupby("case_id")
    daily_prefixes = defaultdict(list)

    for _, case in cases:
        activities = case["activity"].tolist()
        timestamps = case["time:timestamp"].tolist()

        for i in range(1, min(len(activities), prefix_len) + 1):
            prefix = activities[:i]
            event_date = timestamps[i - 1].date()

            encoded_prefix = le.transform(prefix).tolist()
            daily_prefixes[event_date].append(encoded_prefix)

    sorted_dates = sorted(daily_prefixes.keys())

    paired_batches = []

    for i in range(len(sorted_dates) - 1):
        input_date = sorted_dates[i]
        target_date = sorted_dates[i + 1]

        input_sequences = daily_prefixes[input_date]
        target_sequences = daily_prefixes[target_date]

        target_dfg_tensors = [
            sequences_to_sdfa_tensor(target_sequences, num_symbols=len(le.classes_))
        ]
        target_tensor = torch.mean(torch.stack(target_dfg_tensors), dim=0)

        paired_batches.append((input_sequences, target_tensor))

    print(f"Extracted {len(paired_batches)} daily prefixes with shifted targets.")
    print(f"Example input sequence: {paired_batches[0][0][:5]}")  # Print first 5 sequences for verification
    print(f"Example target tensor shape: {paired_batches[0][1].shape}")  # Print shape of target tensor

    return paired_batches


def extract_label_prefixes(df, prefix_len, le):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df['date'] = df['time:timestamp'].dt.date 

    cases = df.groupby("case_id")
    daily_prefixes = {}

    for _, case in cases:
        activities = case["activity"].tolist()
        timestamps = case["time:timestamp"].tolist()

        for i in range(1, min(len(activities), prefix_len) + 1):
            prefix = activities[:i]
            current_event_date = timestamps[i - 1].date()

            encoded_prefix = le.transform(prefix).tolist()

            if current_event_date not in daily_prefixes:
                daily_prefixes[current_event_date] = []

            daily_prefixes[current_event_date].append(encoded_prefix)

    return daily_prefixes


def sequences_to_sdfa_tensor(sequences, num_symbols, pad_token=0, eps=1e-9):
    transition_counts = torch.zeros((num_symbols, num_symbols), dtype=torch.float32)

    for seq in sequences:
        seq = list(seq)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a != pad_token and b != pad_token:
                transition_counts[a - 1, b - 1] += 1.0  # Shift by 1 if pad_token=0

    # de rho-dans
    row_sums = transition_counts.sum(dim=1, keepdim=True) + eps
    sdfa_tensor = transition_counts / row_sums

    # nodig?
    sdfa_tensor[row_sums.squeeze(1) == eps] = 0.0

    return sdfa_tensor