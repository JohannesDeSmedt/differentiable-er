import torch
from tqdm import tqdm
import math

import torch.nn as nn
from preprocessing import EventDatasetTargets
from preprocessing import extract_prefix_suffix_pairs
from torch.utils.data import DataLoader
from torch.nn import functional as F
import editdistance
import time

from create_Seq2Seq import Encoder as LSTM_encoder
from create_Seq2Seq import Decoder as LSTM_decoder
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
    max_len_y = max(len(seq) for seq in ys)

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
        # padded_seq_y = padded_seq_y[:max_len_y]  # truncate if necessary
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
        
        y_emb = self.embedding(y_in) * math.sqrt(self.embedding.embedding_dim)
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
    

class Seq2Seq_trace(nn.Module):
    def __init__(self, num_act, encoder, act_decoder, transformer_mode=False):
        super(Seq2Seq_trace, self).__init__()
        self.num_act = num_act
        self.encoder = encoder
        self.act_decoder = act_decoder
        self.transformer_mode = transformer_mode  # <— flag to switch modes

    def forward(self, trace_prefix_tensor, trace_act_suffix_tensor, teacher_force_ratio=0.5):
        batch_size = trace_prefix_tensor.shape[0]
        memory, hidden, cell = self.encoder(trace_prefix_tensor)

        # If we are using a transformer decoder:
        if self.transformer_mode:
            # Prepare inputs for transformer
            y_in = trace_act_suffix_tensor[:, :-1]  # input sequence without last token
            y_out = trace_act_suffix_tensor[:, 1:]   # target sequence shifted right

            logits = self.act_decoder(
                y_in=y_in,
                memory=memory
            )
            return logits  # shape: (batch, suffix_len-1, num_act)

        # Else use RNN decoder (original logic)
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act, device=trace_prefix_tensor.device)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]
        act_hidden, act_cell = hidden, cell

        for t in range(act_suffix_len):
            act_prediction, act_hidden, act_cell = self.act_decoder(x_act, act_hidden, act_cell)
            act_predictions[:, t, :] = act_prediction
            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0
            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess.float()

        return act_predictions


class SDFA_suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape):
        super().__init__()
        emb_dim = d_model
        self.embedding = nn.Embedding(vocab_size + 1 + 1, emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_dim, 0.1)

        self.encoder = EventTransformer(vocab_size, embedding=self.embedding, pos_encoder=self.pos_encoder, d_model=d_model)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)
        self.suffix_decoder = TransformerSuffixDecoder(d_model, vocab_size, self.embedding, self.pos_encoder)
    
    def forward(self, x, mask, y_in):
        encoded = self.encoder(x, mask)
        sdfa = self.sdfa_proj(encoded)
        
        suffix_logits = self.suffix_decoder(y_in, encoded, memory_key_padding_mask=~mask)

        return sdfa, suffix_logits
    
class suffix_model(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1 + 1, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, 0.1)

        self.encoder = EventTransformer(vocab_size, embedding=self.embedding, pos_encoder=self.pos_encoder, d_model=d_model)
        self.suffix_decoder = TransformerSuffixDecoder(d_model, vocab_size, self.embedding, self.pos_encoder)
    
    def forward(self, x, mask, y_in):
        encoded = self.encoder(x, mask)        
        suffix_logits = self.suffix_decoder(y_in, encoded, memory_key_padding_mask=~mask)

        return suffix_logits
    

class SDFA_suffix_model_LSTM(nn.Module):
    def __init__(self, vocab_size, d_model, sdfa_shape, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, 8, padding_idx=0)

        self.encoder = LSTM_encoder(input_size=8, hidden_size=d_model,
                               num_layers=num_layers, dropout=dropout)
        self.suffix_decoder = LSTM_decoder(input_size=8, hidden_size=d_model,
                               output_size=vocab_size + 2, num_layers=num_layers,
                               dropout=dropout)
        self.sdfa_proj = SDFAProjector(d_model, sdfa_shape)

        hidden_size = d_model
        if hidden_size != d_model:
            self.embedding_to_hidden = nn.Linear(d_model, hidden_size)
        else:
            self.embedding_to_hidden = nn.Identity()

    def forward(self, x, mask, y_in):
        emb_x = self.embedding(x)                          # (B, T_p, d_model)
        emb_x = self.embedding_to_hidden(emb_x)
        enc_states, hidden, cell = self.encoder(emb_x)      # (B, T_p, H), (L, B, H), (L, B, H)

        sdfa = self.sdfa_proj(enc_states)

        emb_y = self.embedding(y_in)                        # (B, T_s, d_model)
        emb_y = self.embedding_to_hidden(emb_y)

        outputs = []
        for t in range(emb_y.size(1)):
            # Feed one token at a time
            pred, hidden, cell = self.suffix_decoder(emb_y[:, t, :], hidden, cell)
            outputs.append(pred.unsqueeze(1))

        suffix_logits = torch.cat(outputs, dim=1)            # (B, T_s, vocab_size+2)

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
                suffix_logits = model(x, mask, y_in)

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

    with torch.no_grad():
        print('Starting evaluation...')
        for dl, (x, mask, y_in, y_out, sdfa_target) in enumerate(dataloader):
            # print('New batch --- ', dl, '/', len(dataloader))
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            # sdfa_pred, suffix_logits = model(x, mask, y_in)
            # plot_sdfa_heatmap(sdfa_pred, sample_idx=0, title="Predicted SDFA (batch 0)")

            if type(model) == SDFA_suffix_model_LSTM:
                embedded = model.embedding(x)
                enc_states, hidden, cell = model.encoder(embedded)
            else:
                memory = model.encoder(x, mask)

            batch_size = x.size(0)
            max_len = y_out.size(1)
            sos_token = y_in[0, 0].item()   # your SOS value
            eoc_index = le.transform(["EOC"])[0]
            # if dl < 5:
            #     print('Max len for generation:', max_len)
            #     print('SOS token:', sos_token, 'EOC token:', eoc_index)

            generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
            for t in range(1, max_len):
                if type(model) == SDFA_suffix_model_LSTM:
                    y_t = model.embedding(generated[:, -1])  # [batch, d_model]

                    out_t, hidden, cell = model.suffix_decoder(y_t, hidden, cell)
                    next_token = out_t.argmax(dim=-1, keepdim=True)

                else:
                    logits = model.suffix_decoder(generated, memory)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                # if dl < 5:
                #     print('Next token at step', t, ':', next_token.squeeze().cpu().tolist())
                #     print('Generated so far:', generated.squeeze().cpu().tolist())
                finished = (generated[:, -1] == eoc_index)
                next_token[finished] = eoc_index
                generated = torch.cat([generated, next_token], dim=1)

                if finished.all():
                    break

            for i in range(batch_size):
                pred_seq = []
                # print('Generated:', generated[i].tolist(), generated[i][1:].tolist())
                for p in generated[i][1:].tolist():
                    pred_seq.append(p)
                    if p == eoc_index:
                        break

                target_seq = []
                for t in y_out[i].tolist():
                    target_seq.append(t)
                    if t == eoc_index:
                        break

                max_len = max(len(pred_seq), len(target_seq))
                if max_len == 0:  # edge case
                    continue
                distance = damerau_levenshtein(pred_seq, target_seq)
                # if dl < 5:
                #     print(f"Sample {i} - DL distance: {distance} (Pred: {pred_seq}, Target: {target_seq}), Max len: {max_len}")
                total_dl_distance += distance / max_len

            # if dl < 5:  # only print first batch to keep output readable
            #     print("\n=== Evaluation Sample ===")
            #     print("Prefix (x):", x[0].tolist())
            #     print("Target Suffix (y_out):", y_out[0].tolist())
            #     print("Predicted Suffix:", generated[0].tolist())
            time.sleep(10)

    avg_dl_distance = total_dl_distance / len(sequences)
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


        # distance = editdistance.eval(pred_seq, true_seq)
        distance = damerau_levenshtein(pred_seq, true_seq)
        total_distance += distance

    avg_distance = total_distance / batch_size
    return avg_distance


def damerau_levenshtein(list1, 
                        list2):

    len_1, len_2 = len(list1), len(list2)

    dist = [[0 for _ in range(len_2 + 1)] for _ in range(len_1 + 1)]

    for i in range(len_1 + 1):
        dist[i][0] = i
    for j in range(len_2 + 1):
        dist[0][j] = j

    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            cost = 0 if list1[i - 1] == list2[j - 1] else 1

            dist[i][j] = min(
                dist[i - 1][j] + 1,    # deletion
                dist[i][j - 1] + 1,    # insertion
                dist[i - 1][j - 1] + cost  # substitution
            )

            if i > 1 and j > 1 and list1[i - 1] == list2[j - 2] and list1[i - 2] == list2[j - 1]:
                dist[i][j] = min(
                    dist[i][j],
                    dist[i - 2][j - 2] + cost  # transposition
                )

    dl_distance = dist[len_1][len_2]

    return dl_distance
    

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

    