import torch
import math

import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class EventTransformer(nn.Module):
    def __init__(self, vocab_size, embedding=None, pos_encoder='auto', d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        else:
            self.embedding = embedding
        if pos_encoder is None:
            self.pos_encoder = None #PositionalEncoding(d_model, dropout)
        elif pos_encoder == 'auto':
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = pos_encoder

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x, mask):
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        if self.pos_encoder is not None:
            embedded = self.pos_encoder(embedded)

        src_key_padding_mask = ~mask.bool()
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        return output

    
class DFGAwareTransformer(nn.Module):
    def __init__(self, vocab_size, embedding=None, pos_encoder=None, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        if embedding is None:
            self.embedding = DFGAwareEmbedding(vocab_size, d_model)
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
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        embedded = self.pos_encoder(embedded)

        src_key_padding_mask = ~mask.bool()
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        return output
    
class AutomatonAwareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        num_states=16,
    ):
        super().__init__()

        self.embedding = DFGAwareEmbedding(vocab_size, d_model)
        self.automaton = SoftAutomaton(num_states, vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x, mask):
        token_emb = self.embedding(x) * (self.d_model ** 0.5)
        state_emb = self.automaton(x)

        embedded = token_emb + state_emb

        if self.pos_encoder is not None:
            embedded = self.pos_encoder(embedded)

        src_key_padding_mask = ~mask.bool()
        output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )
        return output


def pairwise_rank_loss(logits, true_idx, num_neg=5):
    # logits: [B, |A|]
    # true_idx: [B]
    device = logits.device  

    B, A = logits.shape
    loss = 0.0

    for b in range(B):
        pos = logits[b, true_idx[b]]

        # sample negatives
        neg_idx = torch.randperm(A)[:num_neg]
        neg_idx = neg_idx.to(device)  
        neg_idx = neg_idx[neg_idx != true_idx[b]]

        neg = logits[b, neg_idx]
        loss += torch.log1p(torch.exp(neg - pos)).mean()

    return loss / B


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
    

class DFGAwareEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.act_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.future_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)

        nn.init.xavier_uniform_(self.act_emb.weight)
        nn.init.xavier_uniform_(self.future_emb.weight)

    def forward(self, acts):
        return self.act_emb(acts) + self.future_emb(acts)


class SoftAutomaton(nn.Module):

    def __init__(self, num_states, vocab_size, d_model):
        super().__init__()
        self.num_states = num_states

        self.state_emb = nn.Parameter(torch.randn(num_states, d_model))

        # One transition matrix per activity
        self.transitions = nn.Parameter(
            torch.randn(vocab_size + 1 + 1, num_states, num_states)
        )

        nn.init.xavier_uniform_(self.transitions)

    def forward(self, acts, alpha0=None):

        B, T = acts.shape

        if alpha0 is None:
            alpha = torch.zeros(B, self.num_states, device=acts.device)
            alpha[:, 0] = 1.0
        else:
            alpha = alpha0
        # alpha = torch.zeros(B, self.num_states, device=device)
        # alpha[:, 0] = 1.0  # start state

        state_embeddings = []

        for t in range(T):
            a_t = acts[:, t]                       # (B,)
            T_a = self.transitions[a_t]            # (B, S, S)

            alpha = torch.bmm(alpha.unsqueeze(1), T_a).squeeze(1)
            alpha = F.softmax(alpha, dim=-1)

            s_t = alpha @ self.state_emb            # (B, d_model)
            state_embeddings.append(s_t)

        return torch.stack(state_embeddings, dim=1)



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

    s_no_pad = s[:, 1:, 1:]         # remove first row & column (padding)
    L_A_no_pad = L_A[:, 1:, 1:]  
    # fallback_bits_no_pad = fallback_bits[:, 1:, 1:]
    min_dim = min(s_no_pad.size(1), fallback_bits.size(1))
    s_no_pad = s_no_pad[:, :min_dim, :min_dim]
    L_A_no_pad = L_A_no_pad[:, :min_dim, :min_dim]
    # fallback_bits = fallback_bits[:, :min_dim, :min_dim]

    # cost_bits = s_no_pad * (-torch.log2(L_A_no_pad)) + (1 - s_no_pad) * fallback_bits
    try:
        cost_bits = s * (-torch.log2(L_A)) + (1 - s) * fallback_bits
    except:
        print('sdfa_pred shape:', sdfa_pred.shape)
        print('sdfa_target shape:', sdfa_target.shape)
        print('s shape:', s.shape)
        print('L_A shape:', L_A.shape)
        print('rho shape:', rho.shape)
        print('fallback_bits shape:', fallback_bits.shape)


    # cost_bits = s_no_pad * (-torch.log2(L_A_no_pad)) + (1 - s_no_pad) * fallback_bits_no_pad
    avg_cost_bits = torch.mean(cost_bits.view(B, -1), dim=1)  # per batch

    rho_flat = L_A.view(B, -1).clamp(min=eps)
    entropy = -torch.sum(rho_flat * torch.log2(rho_flat), dim=1)

    rel = entropy + avg_cost_bits
    return rel.mean()