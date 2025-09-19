import torch
from tqdm import tqdm
import math

import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

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

class EventDatasetTargets(Dataset):
    def __init__(self, sequences, targets, sdfa_targets, last_token_use, pad_token=0):
        self.sequences = sequences
        self.targets = targets
        self.sdfa_targets = sdfa_targets
        self.pad_token = pad_token
        self.last_token_use = last_token_use

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tgt = self.targets[idx]
        sdfa = self.sdfa_targets[idx]

        if self.last_token_use:
            last_token = tgt[0]
            # last_token = torch.tensor(last_token, dtype=torch.long)
            return seq, last_token, sdfa
        else:
            return seq, tgt, sdfa
    

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


def collate_batch_w_nap_targets(batch, pad_token=0):
    xs, y, sdfas = zip(*batch)  # unpack input/target pairs

    max_len_x = max(len(seq) for seq in xs)
    padded_x, mask = [], []

    for seq_x in xs:
        # Pad input
        padded_seq_x = seq_x + [pad_token] * (max_len_x - len(seq_x))
        padded_x.append(padded_seq_x)
        mask.append([1] * len(seq_x) + [0] * (max_len_x - len(seq_x)))

    x = torch.tensor(padded_x, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    y = torch.tensor(y, dtype=torch.long)
    sdfa_targets = torch.stack(sdfas, dim=0)

    return x, mask, y, sdfa_targets


def collate_batch_w_targets(batch, sos_token=1000, pad_token=0):
    xs, ys, sdfas = zip(*batch)  # unpack input/target pairs

    max_len_x = max(len(seq) for seq in xs)
    max_len_y = max(len(seq) for seq in ys)

    padded_x, mask = [], []
    # padded_y = []
    padded_y_out, padded_y_in = [], []

    for seq_x, seq_y in zip(xs, ys):
        # Pad input
        padded_seq_x = seq_x + [pad_token] * (max_len_x - len(seq_x))
        padded_x.append(padded_seq_x)
        mask.append([1] * len(seq_x) + [0] * (max_len_x - len(seq_x)))

        # Pad target
        padded_seq_y = seq_y + [pad_token] * (max_len_y - len(seq_y))
        padded_y_out.append(padded_seq_y)

        shifted = [sos_token] + padded_seq_y[:-1]
        padded_y_in.append(shifted)
        # padded_y.append(padded_seq_y)

    x = torch.tensor(padded_x, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    # y = torch.tensor(padded_y, dtype=torch.long)
    y_in = torch.tensor(padded_y_in, dtype=torch.long)     # decoder input
    y_out = torch.tensor(padded_y_out, dtype=torch.long)   # decoder target
    sdfa_targets = torch.stack(sdfas, dim=0)

    return x, mask, y_in, y_out, sdfa_targets

def encode_activities(df, le):
    # indien niet in test set -> label -1
    known = set(le.classes_)
    df = df.copy()
    df["activity_encoded"] = df["activity"].apply(lambda x: le.transform([x])[0] if x in known else -1)
    return df


def extract_prefix_suffix_pairs(df, le, length=0, pad_token=0):
    from collections import defaultdict

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    cases = df.groupby("case_id")
    input_sequences, target_sequences, target_dfgs = [], [], []
    paired_batches = []

    for _, case in cases:
        activities = le.transform(case["activity"].tolist()).tolist()
        n = len(activities)

        for i in range(1, n):  # prefix ends at i-1, target starts at i
            if length > 0:  
                prefix = activities[max(0,i-length):i]     
            else:
                prefix = activities[:i]    
            suffix = activities[i:]                            # remaining events (suffix)

            if isinstance(prefix, int):
                prefix = [prefix]
            if isinstance(suffix, int):
                suffix = [suffix]
            input_sequences.append(prefix)
            target_sequences.append(suffix)

    dfgs = sequences_to_sdfa_tensor(target_sequences, num_symbols=len(le.classes_))
    target_dfgs = [dfgs for _ in range(len(input_sequences))]

    print(f"Extracted {len(input_sequences)} prefix/suffix pairs")
    print(f"Example input: {input_sequences[0:2]}")
    print(f"Example target: {target_sequences[1:3]}")

    return input_sequences, target_sequences, target_dfgs


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

        paired_batches.append((input_sequences, target_sequences, target_tensor))

    print(f"Extracted {len(paired_batches)} daily prefixes with shifted targets.")
    print(f"Example input sequence: {paired_batches[0][0][:5]}")  # Print first 5 sequences for verification
    # print(f"Example target tensor shape: {paired_batches[0][1].shape}")  # Print shape of target tensor

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

def sequence_to_sdfa_tensor(sequence, num_symbols, pad_token=0, eps=1e-9):
    transition_counts = torch.zeros((num_symbols, num_symbols), dtype=torch.float32)

    seq = list(sequence)
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


class PaddedLabelEncoder:
    def __init__(self):
        self.le = LabelEncoder()
        
    def fit(self, labels):
        self.le.fit(labels)
        return self
    
    def transform(self, labels):
        return self.le.transform(labels) + 1   # shift
    
    def fit_transform(self, labels):
        return self.le.fit_transform(labels) + 1
    
    def inverse_transform(self, encoded):
        return self.le.inverse_transform(encoded - 1)  # unshift
    
    def len(self):
        return len(self.le.classes_) 
    
    @property
    def classes_(self):
        return self.le.classes_
