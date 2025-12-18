import torch
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import pandas as pd
from preprocessing import EventDatasetTargets
from preprocessing import extract_prefix_suffix_pairs, sequences_to_sdfa_tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time  

from model_help import EventTransformer, SDFAProjector, entropic_relevance_diff_loss
from entropic_relevance import calculate_entropic_relevance
import matplotlib.pyplot as plt


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

def collate_batch_w_local_nap_targets(batch, num_symbols, pad_token=0):
    """
    Collate function that dynamically computes SDFA targets based on the y (targets)
    in the current batch, instead of relying on precomputed global SDFA tensors.
    """
    xs, ys, _ = zip(*batch)  # ignore precomputed sdfas

    # --- Pad the prefix sequences ---
    max_len_x = max(len(seq) for seq in xs)
    padded_x, mask = [], []

    for seq_x in xs:
        padded_seq_x = seq_x + [pad_token] * (max_len_x - len(seq_x))
        padded_x.append(padded_seq_x)
        mask.append([1] * len(seq_x) + [0] * (max_len_x - len(seq_x)))

    # --- Convert to tensors ---
    x = torch.tensor(padded_x, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    y = torch.tensor(ys, dtype=torch.long)  # labels (already numerical)

    y_sequences = [[yi] if isinstance(yi, int) else yi for yi in ys]

    # num_symbols = int(max(x.max().item(), y.max().item()))
    sdfa_targets = sequences_to_sdfa_tensor(y_sequences, num_symbols=num_symbols)

    # Ensure shape is (B, num_symbols, num_symbols)
    if sdfa_targets.dim() == 2:
        sdfa_targets = sdfa_targets.unsqueeze(0)

    return x, mask, y, sdfa_targets


class SingleTokenDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, pooling="mean"):
        super().__init__()
        assert pooling in {"mean", "last", "learnable"}
        self.pooling = pooling
        if pooling == "learnable":
            self.attn = nn.Linear(d_model, 1)   # (B, T, 1)
        self.cls = nn.Linear(d_model, vocab_size + 1)

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
    

def train_NAP_model(dataset, model, le, sequences, optimizer, max_len, er_loss, mix_lambda, device, local=False, num_epochs=10, batch_size=128):
    # model = model.to(device)
    # model.train()

    epoch_time = 0

    ce_losses = []
    er_losses = []
    epoch_loss_er = []
    epoch_loss_ce = []

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        epoch_start = time.perf_counter()  
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le, length=max_len)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, True, pad_token=0)
        if local:
            dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=lambda b: collate_batch_w_local_nap_targets(b, num_symbols=len(le.classes_)))
        else:
            dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=collate_batch_w_nap_targets)

        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Progress", leave=False)
        
        for x, mask, y, sdfa_target in batch_tqdm:  # sequences not needed here
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

            optimizer.zero_grad()

            if er_loss:
                sdfa_pred, logits = model(x, mask)
                # if local:
                    # entropic_loss = entropic_relevance_diff_local_loss(sdfa_pred, sdfa_target)
                # else:
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
                er_losses.append(entropic_loss.item())
            else:
                logits = model(x,mask)
            loss_nap = F.cross_entropy(logits, y, ignore_index=0)
            
            if er_loss:
                loss = (1- mix_lambda) * loss_nap + mix_lambda * entropic_loss
            else:
                loss = loss_nap

            ce_losses.append(loss_nap.item())       

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        epoch_end = time.perf_counter()            # ← 3. end
        epoch_time += epoch_end - epoch_start       # ← 4. elapsed seconds
        epoch_loss_er.append(sum(er_losses) / len(er_losses))
        epoch_loss_ce.append(sum(ce_losses) / len(ce_losses))
        er_losses.clear()
        ce_losses.clear()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(sequences):.4f}")
    
    min_max_normalized_er_losses = [(e - min(epoch_loss_er))/(max(epoch_loss_er)-min(epoch_loss_er)) for e in epoch_loss_er]
    min_max_normalized_ce_losses = [(e - min(epoch_loss_ce))/(max(epoch_loss_ce)-min(epoch_loss_ce)) for e in epoch_loss_ce]

    plt.plot(min_max_normalized_ce_losses, label='Cross-Entropy Loss')
    plt.plot(min_max_normalized_er_losses, label='DIFF-ERO')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Training NAP Losses over Batches')
    plt.legend()
    plt.savefig(f'training_losses_nap_{dataset}.png')
    plt.show()


    return epoch_time / num_epochs


def evaluate_nap_model(model, le, sequences, max_len, er_loss, device, local=False):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le, length=max_len)
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, True, pad_token=0)
    if local:
        dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=lambda b: collate_batch_w_local_nap_targets(b, num_symbols=len(le.classes_)))
    else: 
        dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_batch_w_nap_targets)

    with torch.no_grad():
        for x, mask, y, sdfa_target in dataloader:
            x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)


            if er_loss:
                sdfa_pred, logits = model(x, mask)
                # if local:
                #     entropic_loss = entropic_relevance_diff_local_loss(sdfa_pred, sdfa_target)
                # else:
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)
            else:
                logits = model(x,mask)
            loss_nap = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=-1)
            pred = pred.cpu().numpy()
            y = y.cpu().numpy()

            # if er_loss:
            #     sdfa_pred, _ = model(x, mask, y_in)
            #     batch_entropic_relevance_pred = 0
            #     batch_entropic_relevance_target = 0
            #     for b in range(sdfa_pred.size(0)):
            #         s = sdfa_pred[b]
            #         L_A = s / (s.sum(dim=-1, keepdim=True) + 1e-9)
            #         entropic_relevance_pred = calculate_entropic_relevance(L_A, y_out, le)
            #         entropic_relevance_target = calculate_entropic_relevance(sdfa_target[b], y_out, le)
            #         batch_entropic_relevance_pred += entropic_relevance_pred / len(sdfa_pred)
            #         batch_entropic_relevance_target += entropic_relevance_target / len(sdfa_pred)
            #     total_entropic_relevance_pred += batch_entropic_relevance_pred / len(sdfa_pred)
            #     total_entropic_relevance_target += batch_entropic_relevance_target / len(sdfa_pred)

            #     P = sdfa_pred.clone()
            #     Q = sdfa_target.clone()
            #     print(P.shape, Q.shape)
            #     P /= P.sum(dim=(1, 2), keepdim=True)
            #     Q /= Q.sum(dim=(1, 2), keepdim=True)

            
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