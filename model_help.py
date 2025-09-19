import torch
import math

import torch.nn as nn
from torch.nn import functional as F

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


def entropic_relevance_diff_loss(sdfa_pred, sdfa_target, eps=1e-9):
    B, S, _ = sdfa_pred.shape

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