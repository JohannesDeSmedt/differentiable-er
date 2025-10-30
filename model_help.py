import torch
import math

import torch.nn as nn
import numpy as np
from torch.nn import functional as F

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class EventTransformer(nn.Module):
    def __init__(self, vocab_size, embedding=None, pos_encoder=None, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        else:
            self.embedding = embedding
        if pos_encoder is None:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = pos_encoder

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x, mask):
        embedded = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        embedded = self.pos_encoder(embedded)

        src_key_padding_mask = ~mask.bool()
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        return output
    

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

def entropic_relevance_loss(sdfa_pred, sequences, num_symbols, eps=1e-9):
    # start van sequenties en maakt ground truth SDFA

    B, S, _ = sdfa_pred.shape
    device = sdfa_pred.device

    s = sdfa_pred.clamp(min=eps, max=1 - eps)
    L_A = s / (s.sum(dim=-1, keepdim=True) + eps)

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


def entropic_relevance_diff_local_loss(sdfa_pred, sdfa_target, eps=1e-9):
    B, S, _ = sdfa_pred.shape

    s = sdfa_pred.clamp(min=eps, max=1 - eps)
    L_A = s / (s.sum(dim=-1, keepdim=True) + eps)

    rho = sdfa_target
    fallback_bits = -torch.log2(rho.clamp(min=eps))

    s_no_pad = s[:, :, 1:]        # remove pad column
    L_A_no_pad = L_A[:, :, 1:]
    s_no_pad = s_no_pad[:, 1:, :]
    L_A_no_pad = L_A_no_pad[:, 1:, :]

    # cost_bits = s * (-torch.log2(L_A)) + (1 - s) * fallback_bits
    min_dim = min(s_no_pad.size(1), fallback_bits.size(1))
    s_no_pad = s_no_pad[:, :min_dim, :min_dim]
    L_A_no_pad = L_A_no_pad[:, :min_dim, :min_dim]
    fallback_bits = fallback_bits[:, :min_dim, :min_dim]

    cost_bits = s_no_pad * (-torch.log2(L_A_no_pad)) + (1 - s_no_pad) * fallback_bits
    avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # per batch

    rho_flat = L_A.view(B, -1).clamp(min=eps)
    entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)

    rel = entropy + avg_cost_bits
    return rel.mean()

def entropic_relevance_diff_loss(sdfa_pred, sdfa_target, eps=1e-9):
    B, S, _ = sdfa_pred.shape

    s = sdfa_pred.clamp(min=eps, max=1 - eps)
    L_A = s / (s.sum(dim=-1, keepdim=True) + eps)

    rho = sdfa_target
    fallback_bits = -torch.log2(rho.clamp(min=eps))

    s_no_pad = s[:, 1:, 1:]         # remove first row & column (padding)
    L_A_no_pad = L_A[:, 1:, 1:]  
    # fallback_bits_no_pad = fallback_bits[:, 1:, 1:]
    min_dim = min(s_no_pad.size(1), fallback_bits.size(1))
    s_no_pad = s_no_pad[:, :min_dim, :min_dim]
    L_A_no_pad = L_A_no_pad[:, :min_dim, :min_dim]
    fallback_bits = fallback_bits[:, :min_dim, :min_dim]

    cost_bits = s_no_pad * (-torch.log2(L_A_no_pad)) + (1 - s_no_pad) * fallback_bits

    # cost_bits = s * (-torch.log2(L_A)) + (1 - s) * fallback_bits
    # cost_bits = s_no_pad * (-torch.log2(L_A_no_pad)) + (1 - s_no_pad) * fallback_bits_no_pad
    avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # per batch

    rho_flat = L_A.view(B, -1).clamp(min=eps)
    entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)

    rel = entropy + avg_cost_bits
    return rel.mean()