import torch
from tqdm import tqdm
import math

import numpy as np
import torch.nn as nn
import pandas as pd
from preprocessing import sequences_to_sdfa_tensor, EventDataset, collate_batch
from torch.utils.data import DataLoader


device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class EventTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x, mask):
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        embedded = self.pos_encoder(embedded)
        src_key_padding_mask = ~mask.bool()
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        return output


class SuffixDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, encoded):
        out, _ = self.decoder(encoded)
        logits = self.output(out)
        return logits  # (batch_size, seq_len, vocab_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SDFAProjector(nn.Module):
    def __init__(self, d_model, sdfa_shape):
        super().__init__()
        self.proj = nn.Linear(d_model, int(np.prod(sdfa_shape)))
        self.sdfa_shape = sdfa_shape

    def forward(self, encoded):
        pooled = encoded.mean(dim=1)
        out = self.proj(pooled)
        soft_tensor = torch.sigmoid(out).view(-1, *self.sdfa_shape)
        return soft_tensor


def sdfa_loss(pred, target, eps=1e-9):
    if pred.shape != target.shape:
        # Try to match batch dim
        if target.shape[0] == 1 and pred.shape[0] > 1:
            target = target.expand_as(pred)
        else:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    pred = pred.clamp(min=eps, max=1 - eps)
    return -torch.mean(target * torch.log2(pred) + (1 - target) * torch.log2(1 - pred))


def entropic_relevance_loss(sdfa_pred, sequences, num_symbols, eps=1e-9):

    B, S, _ = sdfa_pred.shape
    device = sdfa_pred.device

    s = sdfa_pred.clamp(min=eps, max=1 - eps)
    print(s.shape)

    L_A = s / (s.sum(dim=-1, keepdim=True) + eps)
    print(L_A.shape)

    freq_tensor = torch.zeros((B, num_symbols, num_symbols), device=device)
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - 1):
            a, b = seq[j], seq[j + 1]
            freq_tensor[i, a, b] += 1.0
    rho = freq_tensor / (freq_tensor.sum(dim=(1, 2), keepdim=True) + eps)
    fallback_bits = -torch.log2(rho.clamp(min=eps))

    cost_bits = s * (-torch.log2(L_A)) + (1 - s) * fallback_bits
    avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # per batch

    rho_flat = L_A.view(B, -1).clamp(min=eps)
    entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)

    rel = entropy + avg_cost_bits
    return rel.mean()


def entropic_relevance_diff_loss(sdfa_pred, sdfa_target, eps=1e-9):

    B, S, _ = sdfa_pred.shape
    device = sdfa_pred.device

    s = sdfa_pred.clamp(min=eps, max=1 - eps)

    L_A = s / (s.sum(dim=-1, keepdim=True) + eps)

    rho = sdfa_target
    fallback_bits = -torch.log2(rho.clamp(min=eps))

    cost_bits = s * (-torch.log2(L_A)) + (1 - s) * fallback_bits
    avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # per batch

    rho_flat = L_A.view(B, -1).clamp(min=eps)
    entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)

    rel = entropy + avg_cost_bits
    return rel.mean()


# def entropic_relevance_loss(sdfa_pred, sequences, num_symbols, eps=1e-9):
#     B = sdfa_pred.size(0)  # batch size
#     device = sdfa_pred.device

#     freq_tensor = torch.zeros((B, num_symbols, num_symbols), device=device)
#     for i, seq in enumerate(sequences):
#         for j in range(len(seq) - 1):
#             a, b = seq[j], seq[j + 1]
#             freq_tensor[i, a, b] += 1.0

#     rho = freq_tensor / (freq_tensor.sum(dim=(1, 2), keepdim=True) + eps)

#     rho_flat = rho.view(B, -1).clamp(min=eps)
#     entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)  # shape (B,)

#     # s(t, A) = sdfa_pred
#     # L_A(t) = sdfa_pred
#     # bits(t, E, A) = -log₂ (rho), as fallback
#     soft_membership = sdfa_pred.clamp(min=eps, max=1 - eps)
#     fallback_bits = (-torch.log2(rho.clamp(min=eps))).detach()  # stop gradients here

#     # cost_bits(t, E, A) = s ⋅ (-log₂ L_A(t)) + (1 - s) ⋅ bits(t, E, A)
#     cost_bits = soft_membership * (-torch.log2(soft_membership)) + \
#                 (1 - soft_membership) * fallback_bits

#     avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # shape (B,)

#     rel = entropy + avg_cost_bits

#     return rel.mean()


# def entropic_relevance_diff_loss(sdfa_pred, sdfa_target, eps=1e-9):
#     # Clamp to avoid log(0)
#     pred = sdfa_pred.clamp(min=eps, max=1 - eps)

#     # (s) and (1 - s)
#     s = sdfa_target
#     one_minus_s = 1 - s

#     log_term = -torch.log2(pred)
#     bits_term = -torch.log2(1 - pred)

#     cost_bits = s * log_term + one_minus_s * bits_term
#     cost_bits_mean = torch.mean(cost_bits, dim=(1, 2))  # mean over matrix

#     pred_sum = pred.sum(dim=(1, 2), keepdim=True)
#     rho = pred / (pred_sum + eps)
#     H0 = -torch.sum(rho * torch.log2(rho + eps), dim=(1, 2))  # entropy per batch

#     rel = H0 + cost_bits_mean
#     return rel.mean()




class SDFA_suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)
        self.suffix_decoder = SuffixDecoder(d_model, vocab_size)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        sdfa = self.sdfa_proj(encoded)
        suffix_logits = self.suffix_decoder(encoded)
        return sdfa, suffix_logits

class SDFA_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.projector = SDFAProjector(d_model, sdfa_shape)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        soft_tensor = self.projector(encoded)
        return soft_tensor


def train_suffix_model(model, le, daily_sequences, optimizer, max_len, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        for day_idx, (sequences, target_dfg_tensor) in enumerate(daily_sequences):  # list of (X, y)
            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)

            batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Day {day_idx+1} Progress", leave=False)
            

            for x, mask, _ in batch_tqdm:  # sequences not needed here
                x, mask = x.to(device), mask.to(device)

                # Repeat target for each item in batch

                optimizer.zero_grad()
                sdfa_pred = model(x, mask)
                # visualize_sdfa(sdfa_pred, symbol_labels=le.classes_, title="Predicted SDFA", idx=0)
                # for i in range(min(4, sdfa_pred.shape[0])):
                #     visualize_sdfa(sdfa_pred, symbol_labels=le.classes_, title="Predicted SDFA", idx=i)

                logic = decoder(encoder_outputs)

                sdfa_target = target_dfg_tensor.to(device).unsqueeze(0).expand(sdfa_pred.shape[0], -1, -1)
                # sdfa_pred_last = sdfa_pred[:, -1, :, :]  # Get the last time step prediction

                # loss = sdfa_loss(sdfa_pred, sdfa_target)
                loss_suffix = F.cross_entropy(suffix_logits.view(-1, len(le)), y_suffix.view(-1), ignore_index=pad_token)
                loss = entropic_relevance_loss(sdfa_pred, sequences, num_symbols=len(le.classes_))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(daily_sequences):.4f}")


def train_model(model, le, daily_sequences, optimizer, max_len, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        for day_idx, (sequences, target_dfg_tensor) in enumerate(daily_sequences):  # list of (X, y)
            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)

            batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Day {day_idx+1} Progress", leave=False)
            

            for x, mask, _ in batch_tqdm:  # sequences not needed here
                x, mask = x.to(device), mask.to(device)

                # Repeat target for each item in batch

                optimizer.zero_grad()
                sdfa_pred = model(x, mask)
                # visualize_sdfa(sdfa_pred, symbol_labels=le.classes_, title="Predicted SDFA", idx=0)
                # for i in range(min(4, sdfa_pred.shape[0])):
                #     visualize_sdfa(sdfa_pred, symbol_labels=le.classes_, title="Predicted SDFA", idx=i)



                sdfa_target = target_dfg_tensor.to(device).unsqueeze(0).expand(sdfa_pred.shape[0], -1, -1)
                # sdfa_pred_last = sdfa_pred[:, -1, :, :]  # Get the last time step prediction

                # loss = sdfa_loss(sdfa_pred, sdfa_target)
                loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)#sequences, num_symbols=len(le.classes_))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(daily_sequences):.4f}")


def evaluate_model(model, le, daily_sequences, max_len):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():

        for day_idx, (sequences, target_dfg_tensor) in enumerate(daily_sequences):  # list of (X, y)
            train_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_batch)

            test_dataset = EventDataset(sequences, pad_token=0)
            dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_batch)

            for x, mask, sequences in dataloader:
                x, mask = x.to(device), mask.to(device)

                sdfa_pred = model(x, mask)
                sdfa_target = target_dfg_tensor.to(device).unsqueeze(0).expand(sdfa_pred.shape[0], -1, -1)

                # loss = sdfa_loss(sdfa_pred, sdfa_target)
                loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
                # loss = entropic_relevance_loss(sdfa_pred, sequences, num_symbols=len(le.classes_))
                total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")



# def sdfa_to_dfg(sdfa_tensor, le, threshold=0.5):
#     dfg = {}
#     sigma = le.classes_

#     sdfa_collapsed = sdfa_tensor.sum(dim=(0, 1)) 

#     for i in range(sdfa_collapsed.shape[0]):
#         for j in range(sdfa_collapsed.shape[1]):
#             weight = sdfa_collapsed[i, j].item()
#             if weight >= threshold:
#                 from_act = sigma[i]
#                 to_act = sigma[j]
#                 dfg_key = (from_act, to_act)
#                 dfg[dfg_key] = weight

#     return dfg


    
# import matplotlib.pyplot as plt
# import seaborn as sns


# def visualize_sdfa(sdfa_tensor, symbol_labels=None, title="Soft SDFA", idx=0):
#     """
#     Visualize a soft SDFA transition matrix as a heatmap.
    
#     Args:
#         sdfa_tensor: torch.Tensor of shape (A, A) or (B, A, A)
#         symbol_labels: list of A symbol names (optional)
#         title: plot title
#         idx: index if sdfa_tensor is a batch
#     """
#     if sdfa_tensor.ndim == 3:
#         sdfa = sdfa_tensor[idx].detach().cpu().numpy()
#     else:
#         sdfa = sdfa_tensor.detach().cpu().numpy()

#     A = sdfa.shape[0]
#     if symbol_labels is None:
#         symbol_labels = [str(i) for i in range(A)]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(sdfa, xticklabels=symbol_labels, yticklabels=symbol_labels,
#                 cmap="viridis", annot=True, fmt=".2f", cbar=True)

#     plt.xlabel("To Symbol")
#     plt.ylabel("From Symbol")
#     plt.title(f"{title} [{idx}]")
#     plt.tight_layout()
#     plt.show()
