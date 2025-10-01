import torch
from tqdm import tqdm
import math

import torch.nn as nn
from preprocessing import EventDatasetTargets
from preprocessing import extract_prefix_suffix_pairs
from torch.utils.data import DataLoader
from torch.nn import functional as F
import editdistance

from model_help import EventTransformer, SDFAProjector, PositionalEncoding, entropic_relevance_diff_loss

# def collate_batch_w_targets(batch, pad_token=0):
#     """
#     batch = [(prefix, suffix, sdfa_target), ...]
#     Produces:
#         x      : padded prefixes
#         mask   : prefix attention mask
#         y_in   : padded decoder input (shifted suffix)
#         y_out  : padded decoder target
#         sdfa   : stacked sdfa_targets
#     """
#     prefixes, suffixes, sdfa_targets = zip(*batch)

#     # Pad prefixes
#     x_lens = [len(p) for p in prefixes]
#     max_x_len = max(x_lens)

#     x = torch.full((len(prefixes), max_x_len), pad_token, dtype=torch.long)
#     for i, seq in enumerate(prefixes):
#         x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

#     mask = (x != pad_token)

#     s_lens = [len(s) for s in suffixes]
#     max_s_len = max(s_lens)
#     y_in  = torch.full((len(suffixes), max_s_len), pad_token, dtype=torch.long)
#     y_out = torch.full((len(suffixes), max_s_len), pad_token, dtype=torch.long)

#     for i, seq in enumerate(suffixes):
#         seq_t = torch.tensor(seq, dtype=torch.long)
#         # y_in gets everything except last token
#         y_in[i, :len(seq)-1]  = seq_t[:-1]
#         # y_out gets everything except first token
#         y_out[i, :len(seq)-1] = seq_t[1:]

#     # SDFA targets -> stack into tensor
#     sdfa_target = torch.stack(sdfa_targets)

#     return x, mask, y_in, y_out, sdfa_target


def collate_batch_w_targets(batch, max_case_len, sos_token=1000, pad_token=0):
    xs, ys, sdfas = zip(*batch)  # unpack input/target pairs

    max_len_x = max(len(seq) for seq in xs)
    max_len_y = max_case_len #max(len(seq) for seq in ys)

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
        padded_seq_y = padded_seq_y[:max_len_y]  # truncate if necessary
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


def train_suffix_model(model, le, sequences, optimizer, max_case_len, er_loss, mix_lambda, device, num_epochs=10):
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False,
                                 collate_fn=lambda b: collate_batch_w_targets(b, max_case_len=max_case_len, sos_token=le.len()+1))

        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Progress", leave=False)
        
        for batch_idx, (x, mask, y_in, y_out, sdfa_target) in enumerate(batch_tqdm):
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            if batch_idx == 0:  # only print first batch to keep output readable
                print("\n--- Debug: Batch 0 ---")
                print("x (prefixes, padded):")
                print(x[0:3].cpu().tolist())  # first 3 samples

                print("\nmask (True=real token, False=pad):")
                print(mask[0:3].cpu().tolist())

                print("\ny_in (decoder input, shifted with SOS):")
                print(y_in[0:3].cpu().tolist())

                print("\ny_out (decoder target, suffix padded):")
                print(y_out[0:3].cpu().tolist())

                print("\nsdfa_target shape:", tuple(sdfa_target.shape))
                print("--- End Debug ---\n")

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


def evaluate_suffix_model(model, le, sequences, max_case_len, er_loss, device):
    model.eval()
    total_loss = 0.0
    total_dl_distance = 0.0

    prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            collate_fn=lambda b: collate_batch_w_targets(b, max_case_len=max_case_len, sos_token=le.len()+1))


    end_token = le.transform(["End"])[0]
    start_token = le.transform(["Start"])[0]
    with torch.no_grad():
        print('Starting evaluation...')
        for dl, (x, mask, y_in, y_out, sdfa_target) in enumerate(dataloader):
            # print('New batch --- ', dl, '/', len(dataloader))
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            # sdfa_pred, suffix_logits = model(x, mask, y_in)
            # plot_sdfa_heatmap(sdfa_pred, sample_idx=0, title="Predicted SDFA (batch 0)")

            memory = model.encoder(x, mask)
            # print('Encoded')

            batch_size = x.size(0)
            max_len = y_out.size(1)
            sos_token = y_in[0, 0].item()   # your SOS value

            generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
            # print('Starting generation')
            # print('Max len is ', max_len)
            for t in range(1, max_len):
                # print('T is ', t, ' / ', max_len)
                logits = model.suffix_decoder(generated, memory, 
                                            memory_key_padding_mask=~mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if (next_token.squeeze(-1) == end_token).all():
                    break
                
            if dl < 5:  # only print first batch to keep output readable
                print("\n=== Evaluation Sample ===")
                print("Prefix (x):", x[0].tolist())
                print("Target Suffix (y_out):", y_out[0].tolist())
                print("Predicted Suffix:", generated[0].tolist())

            generated_no_sos = generated[:, 1:]
            batch_distance = compute_avg_damerau_levenshtein(generated_no_sos, y_out)
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

        # print("\nPredicted Suffix (no pad):", pred_seq)
        # print("True Suffix (no pad):", true_seq)

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



class TransformerDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size + 1)

    def forward(self, encoded):
        seq_len = encoded.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(encoded.device)
        out = self.transformer_decoder(tgt=encoded, memory=encoded, tgt_mask=causal_mask)
        logits = self.output(out)
        return logits


class suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        self.encoder = EventTransformer(vocab_size, d_model=d_model)
        self.suffix_decoder = TransformerDecoder(d_model, vocab_size)

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        suffix_logits = self.suffix_decoder(encoded)
        return suffix_logits
    

class SuffixDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size + 1)

    def forward(self, encoded):
        out, _ = self.decoder(encoded)
        logits = self.output(out)
        return logits  # (batch_size, seq_len, vocab_size)
    

import matplotlib.pyplot as plt
import seaborn as sns

def plot_sdfa_heatmap(sdfa_tensor, sample_idx=0, title="SDFA Heatmap"):
    """
    Plots a heatmap of the SDFA tensor for a given sample.
    
    Args:
        sdfa_tensor: torch.Tensor of shape (batch, *sdfa_shape)
        sample_idx: index in the batch to visualize
        title: plot title
    """
    # detach and move to CPU
    sdfa_np = sdfa_tensor[sample_idx].detach().cpu().numpy()

    # if it's higher than 2D, flatten last dims into 2D
    if sdfa_np.ndim > 2:
        sdfa_np = sdfa_np.reshape(sdfa_np.shape[0], -1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(sdfa_np, vmin=0, vmax=1, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Target symbol")
    plt.ylabel("Source symbol")
    plt.show()

    