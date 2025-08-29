import torch
from tqdm import tqdm
import math

import numpy as np
import torch.nn as nn
import pandas as pd
from preprocessing import EventDataset, collate_batch, EventDatasetTargets, collate_batch_w_targets, collate_batch_w_nap_targets
from preprocessing import extract_prefix_suffix_pairs
from torch.utils.data import DataLoader
from torch.nn import functional as F
import editdistance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


class SingleTokenDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, pooling="mean"):
        super().__init__()
        assert pooling in {"mean", "last", "learnable"}
        self.pooling = pooling
        if pooling == "learnable":
            self.attn = nn.Linear(d_model, 1)   # (B, T, 1)
        self.cls = nn.Linear(d_model, vocab_size)

    def forward(self, encoded):
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)   # (B, d_model)
        elif self.pooling == "last":
            pooled = encoded[:, -1]        # (B, d_model)
        else: 
            attn_weights = torch.softmax(self.attn(encoded), dim=1)  # (B, T, 1)
            pooled = torch.sum(encoded * attn_weights, dim=1)      # (B, d_model)

        logits = self.cls(pooled)          # (B, vocab_size)
        return logits
    

class SuffixDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, encoded):
        out, _ = self.decoder(encoded)
        logits = self.output(out)
        return logits  # (batch_size, seq_len, vocab_size)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, encoded):
        seq_len = encoded.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(encoded.device)
        out = self.transformer_decoder(tgt=encoded, memory=encoded, tgt_mask=causal_mask)
        logits = self.output(out)
        return logits


class SDFA_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.projector = SDFAProjector(d_model, sdfa_shape)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        soft_tensor = self.projector(encoded)
        return soft_tensor
    

class SDFA_NAP_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)
        self.item_decoder = SingleTokenDecoder(d_model, vocab_size, pooling='mean')

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        sdfa = self.sdfa_proj(encoded)
        logits = self.item_decoder(encoded)
        return sdfa, logits
    

class NAP_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.item_decoder = SingleTokenDecoder(d_model, vocab_size, pooling='mean')

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        logits = self.item_decoder(encoded)
        return logits


class SDFA_suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)
        # self.suffix_decoder = SuffixDecoder(d_model, vocab_size)
        self.suffix_decoder = TransformerDecoder(d_model, vocab_size)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        sdfa = self.sdfa_proj(encoded)
        suffix_logits = self.suffix_decoder(encoded)
        return sdfa, suffix_logits


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


def train_model(model, le, daily_sequences, optimizer, max_len, num_epochs=10):
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



def evaluate_model(model, le, daily_sequences, max_len):
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


def train_NAP_model(model, le, sequences, optimizer, max_len, er_loss, mix_lambda, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, True, pad_token=0)
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch_w_nap_targets)

        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Progress", leave=False)
        
        for x, mask, y, sdfa_target in batch_tqdm:  # sequences not needed here
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

            optimizer.zero_grad()

            if er_loss:
                sdfa_pred, logits = model(x, mask)
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
            else:
                logits = model(x,mask)
            loss_nap = F.cross_entropy(logits, y)
            
            if er_loss:
                loss = loss_nap + mix_lambda * entropic_loss
            else:
                loss = loss_nap
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(sequences):.4f}")

def evaluate_nap_model(model, le, sequences, max_len, er_loss):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, True, pad_token=0)
    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_batch_w_nap_targets)

    with torch.no_grad():
        for x, mask, y, sdfa_target in dataloader:
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)


            if er_loss:
                sdfa_pred, logits = model(x, mask)
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
            else:
                logits = model(x,mask)
            loss_nap = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=-1)
            
            if er_loss:
                loss = loss_nap + 0.5 * entropic_loss
            else:
                loss = loss_nap
            total_loss += loss.item()

            accuracy = accuracy_score(pred, y)
            precision = precision_score(pred, y, average='weighted')
            recall = recall_score(pred, y, average='weighted')
            f1 = f1_score(pred, y, average='weighted')
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

    avg_accuracy = total_accuracy / len(dataloader)
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader) 
    print(f"Avg Accuracy on test set: {avg_accuracy:.4f}")
    print(f"Avg Precision on test set: {avg_precision:.4f}")
    print(f"Avg Recall on test set: {avg_recall:.4f}")
    print(f"Avg F1-score on test set: {avg_f1:.4f}")
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")
    return total_loss/len(dataloader), avg_accuracy, avg_precision, avg_recall, avg_f1


def train_suffix_model(model, le, sequences, optimizer, max_len, er_loss, mix_lambda, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch_w_targets)

        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Progress", leave=False)
        
        for x, mask, y, sdfa_target in batch_tqdm:  # sequences not needed here
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

            optimizer.zero_grad()
            sdfa_pred, suffix_logits = model(x, mask)

            entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)

            seq_len_pred = suffix_logits.size(1)
            seq_len_target = y.size(1)
            min_len = min(seq_len_pred, seq_len_target)

            loss_suffix = F.cross_entropy(
                suffix_logits[:, :min_len, :].reshape(-1, suffix_logits.size(-1)),
                y[:, :min_len].reshape(-1),
                ignore_index=0,
            )
            
            if er_loss:
                loss = loss_suffix + mix_lambda * entropic_loss
            else:
                loss = loss_suffix
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(sequences):.4f}")


def evaluate_suffix_model(model, le, sequences, max_len, er_loss):
    model.eval()
    total_loss = 0.0
    total_dl_distance = 0.0

    prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, pad_token=0)
    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_batch_w_targets)

    with torch.no_grad():
        for x, mask, y, sdfa_target in dataloader:
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

            sdfa_pred, suffix_logits = model(x, mask)

            entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)

            seq_len_pred = suffix_logits.size(1)
            seq_len_target = y.size(1)
            min_len = min(seq_len_pred, seq_len_target)
            suffix_pred = suffix_logits.argmax(dim=-1)

            loss_suffix = F.cross_entropy(
                suffix_logits[:, :min_len, :].reshape(-1, suffix_logits.size(-1)),
                y[:, :min_len].reshape(-1),
                ignore_index=0,
            )
            
            if er_loss:
                loss = loss_suffix + 0.5 * entropic_loss
            else:
                loss = loss_suffix
            total_loss += loss.item()

            batch_distance = compute_avg_damerau_levenshtein(suffix_pred, y)
            total_dl_distance += batch_distance

    avg_dl_distance = total_dl_distance / len(dataloader)
    print(f"Avg Damerau-Levenshtein distance on test set: {avg_dl_distance:.4f}")
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")
    return total_loss/len(dataloader), avg_dl_distance


def compute_avg_damerau_levenshtein(suffix_pred, suffix_true):
    batch_size = suffix_pred.size(0)
    total_distance = 0.0

    suffix_pred_np = suffix_pred.cpu().numpy()
    suffix_true_np = suffix_true.cpu().numpy()

    for i in range(batch_size):
        # Remove padding (0)
        pred_seq = [x for x in suffix_pred_np[i] if x != 0]
        true_seq = [x for x in suffix_true_np[i] if x != 0]

        distance = editdistance.eval(pred_seq, true_seq)
        total_distance += distance

    avg_distance = total_distance / batch_size
    return avg_distance


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

# def sdfa_loss(pred, target, eps=1e-9):
#     if pred.shape != target.shape:
#         # Try to match batch dim
#         if target.shape[0] == 1 and pred.shape[0] > 1:
#             target = target.expand_as(pred)
#         else:
#             raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

#     pred = pred.clamp(min=eps, max=1 - eps)
#     return -torch.mean(target * torch.log2(pred) + (1 - target) * torch.log2(1 - pred))
