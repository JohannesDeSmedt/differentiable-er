import torch
from tqdm import tqdm
import math

import torch.nn as nn
from preprocessing import EventDatasetTargets, collate_batch_w_targets
from preprocessing import extract_prefix_suffix_pairs
from torch.utils.data import DataLoader
from torch.nn import functional as F
import editdistance

from model_help import PositionalEncoding, entropic_relevance_diff_loss

class SuffixDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size + 1)

    def forward(self, encoded):
        out, _ = self.decoder(encoded)
        logits = self.output(out)
        return logits  # (batch_size, seq_len, vocab_size)


class TransformerSuffixDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, embedding, pos_encoder, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = embedding #nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_encoding = pos_encoder #PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size + 1)

    def forward(self, y_in, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        y_emb = self.embedding(y_in) * math.sqrt(self.d_model)
        y_emb = self.pos_encoding(y_emb)

        if tgt_mask is None:
            tgt_len = y_in.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(y_in.device)

        out = self.transformer_decoder(
            tgt=y_emb, memory=memory, tgt_mask=tgt_mask,memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output(out) 
        return logits

class SDFA_suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1 + 1, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, 0.1)

        self.encoder = EventTransformer(vocab_size, embedding=self.embedding, pos_encoder=self.pos_encoder, d_model=d_model)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)
        self.suffix_decoder = TransformerSuffixDecoder(d_model, vocab_size, self.embedding, self.pos_encoder)
    
    def forward(self, x, mask, y_in):
        encoded = self.encoder(x, mask)
        sdfa = self.sdfa_proj(encoded)
        suffix_logits = self.suffix_decoder(y_in, encoded, memory_key_padding_mask=~mask)

        return sdfa, suffix_logits
    

def train_suffix_model(model, le, sequences, optimizer, max_len, er_loss, mix_lambda, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=lambda b: collate_batch_w_targets(b, sos_token=le.len()+1))

        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Progress", leave=False)
        
        # for x, mask, y, sdfa_target in batch_tqdm:  # sequences not needed here
        #     x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

        #     optimizer.zero_grad()

        for x, mask, y_in, y_out, sdfa_target in batch_tqdm:
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            optimizer.zero_grad()

            # sdfa_pred, suffix_logits = model(x, mask, y_in)

            if er_loss:
                sdfa_pred, suffix_logits = model(x, mask, y_in)
            else:
                suffix_logits = model(x, mask)

            if er_loss:
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)

            seq_len_pred = suffix_logits.size(1)
            seq_len_target = y_out.size(1)
            min_len = min(seq_len_pred, seq_len_target)

            loss_suffix = F.cross_entropy(suffix_logits[:, :min_len, :].reshape(-1, suffix_logits.size(-1)),
                y_out[:, :min_len].reshape(-1), ignore_index=0,
            )
            
            if er_loss:
                loss = (1-mix_lambda) * loss_suffix + mix_lambda * entropic_loss
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
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=lambda b: collate_batch_w_targets(b, sos_token=le.len()+1))

    with torch.no_grad():
        # for x, mask, y, sdfa_target in dataloader:
        #     x, mask, y, sdfa_target = x.to(device), mask.to(device), y.to(device), sdfa_target.to(device)

        #     if er_loss:
        #         sdfa_pred, suffix_logits = model(x, mask)
        #     else:
        #         suffix_logits = model(x, mask)
        for x, mask, y_in, y_out, sdfa_target in dataloader:
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            if er_loss:
                sdfa_pred, suffix_logits = model(x, mask, y_in)
            else:
                suffix_logits = model(x, mask)

            if er_loss:
                entropic_loss = entropic_relevance_diff_loss(sdfa_pred, sdfa_target)

            seq_len_pred = suffix_logits.size(1)
            seq_len_target = y_out.size(1)
            min_len = min(seq_len_pred, seq_len_target)
            suffix_pred = suffix_logits.argmax(dim=-1)

            loss_suffix = F.cross_entropy(suffix_logits[:, :min_len, :].reshape(-1, suffix_logits.size(-1)),
                y_out[:, :min_len].reshape(-1), ignore_index=0,
            )
            
            if er_loss:
                loss = loss_suffix + 0.5 * entropic_loss
            else:
                loss = loss_suffix
            total_loss += loss.item()

            batch_distance = compute_avg_damerau_levenshtein(suffix_pred, y_out)
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