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
    

def encode_activities(df, le):
    # indien niet in test set -> label -1
    known = set(le.classes_)
    df = df.copy()
    df["activity_encoded"] = df["activity"].apply(lambda x: le.transform([x])[0] if x in known else -1)
    return df

def encode_activities_with_dict(df, my_dict):
    # indien niet in test set -> label -1
    df = df.copy()
    df["concept:name"] = df["concept:name"].apply(lambda x: my_dict[x] if x in my_dict.keys() else 1)
    return df


def extract_prefix_suffix_pairs(df, le, length=0, pad_token=0):
    from collections import defaultdict

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'],format='mixed')

    cases = df.groupby("case_id")
    input_sequences, target_sequences, target_dfgs = [], [], []

    for _, case in cases:
        activities = le.transform(case["activity"].tolist()).tolist()
        n = len(activities)

        for i in range(1, n):  # prefix ends at i-1, target starts at i
            # if len(activities[:i]) < 10:
                # continue

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
    def __init__(self, unk_token='UNK'):
        self.le = LabelEncoder()
        self.unk_token = unk_token
        
    def fit(self, labels):
        self.le.fit(labels)
        return self
    
    def transform(self, labels):        
        labels = np.array(labels, dtype=str)
        known_classes = set(self.le.classes_)

        encoded = []
        for lbl in labels:
            if lbl in known_classes:
                encoded.append(self.le.transform([lbl])[0] + 1)
            else:
                # unseen → last index (len(classes_) + 1, since +1 shift is used)
                encoded.append(len(self.le.classes_) + 1)
        return np.array(encoded)
    
    # def transform(self, labels):
    #     return self.le.transform(labels) + 1   # shift
    
    def fit_transform(self, labels):
        return self.le.fit_transform(labels) + 1
    
    def inverse_transform(self, encoded):
        return self.le.inverse_transform(encoded - 1)  # unshift
    
    def len(self):
        return len(self.le.classes_) + 1
    
    @property
    def classes_(self):
        # Insert both PAD and UNK into vocab for clarity
        return np.insert(self.le.classes_, [len(self.le.classes_)], [self.unk_token])

    # @property
    # def classes_(self):
    #     return self.le.classes_
