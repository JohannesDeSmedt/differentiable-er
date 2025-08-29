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

class EventDatasetTargets(Dataset):
    def __init__(self, sequences, targets, sdfa_targets, last_token_use, pad_token=0):
        self.sequences = sequences
        self.targets = targets
        self.sdfa_targets = sdfa_targets
        self.pad_token = pad_token
        self.last_token_use = last_token_use

        # print('Sequenties:')
        # print(sequences[0:5])
        # print('Targets:')
        # print(targets[0:5])


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tgt = self.targets[idx]
        sdfa = self.sdfa_targets[idx]
        # print('In getitem:')
        # print('Seq:')
        # print(seq)
        # print('Tgt:')
        # print(tgt[0])
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


def collate_batch_w_targets(batch, pad_token=0):
    xs, ys, sdfas = zip(*batch)  # unpack input/target pairs

    max_len_x = max(len(seq) for seq in xs)
    max_len_y = max(len(seq) for seq in ys)

    padded_x, mask = [], []
    padded_y = []

    for seq_x, seq_y in zip(xs, ys):
        # Pad input
        padded_seq_x = seq_x + [pad_token] * (max_len_x - len(seq_x))
        padded_x.append(padded_seq_x)
        mask.append([1] * len(seq_x) + [0] * (max_len_x - len(seq_x)))

        # Pad target
        padded_seq_y = seq_y + [pad_token] * (max_len_y - len(seq_y))
        padded_y.append(padded_seq_y)

    x = torch.tensor(padded_x, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    y = torch.tensor(padded_y, dtype=torch.long)
    sdfa_targets = torch.stack(sdfas, dim=0)

    return x, mask, y, sdfa_targets

def encode_activities(df, le):
    # indien niet in test set -> label -1
    known = set(le.classes_)
    df = df.copy()
    df["activity_encoded"] = df["activity"].apply(lambda x: le.transform([x])[0] if x in known else -1)
    return df


def extract_prefix_suffix_pairs(df, le, pad_token=0):
    from collections import defaultdict

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    cases = df.groupby("case_id")
    input_sequences, target_sequences, target_dfgs = [], [], []
    paired_batches = []

    for _, case in cases:
        activities = le.transform(case["activity"].tolist()).tolist()
        n = len(activities)

        for i in range(1, n):  # prefix ends at i-1, target starts at i
            prefix = activities[:i]             # input prefix
            suffix = activities[i:]             # remaining events (suffix)

            if isinstance(prefix, int):
                prefix = [prefix]
            if isinstance(suffix, int):
                suffix = [suffix]
            input_sequences.append(prefix)
            target_sequences.append(suffix)
            target_dfg_tensor = sequence_to_sdfa_tensor(suffix, num_symbols=len(le.classes_))          
            # target_tensor = torch.mean(torch.stack(target_dfg_tensor), dim=0)
            target_dfgs.append(target_dfg_tensor)
            paired_batches.append((prefix, suffix, target_dfg_tensor))

    print(f"Extracted {len(paired_batches)} prefix/suffix pairs")
    print(f"Example input: {paired_batches[0][0]}")
    print(f"Example target: {paired_batches[0][1]}")

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