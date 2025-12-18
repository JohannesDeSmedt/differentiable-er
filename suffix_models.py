import torch
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch.nn as nn
from preprocessing import EventDatasetTargets
from preprocessing import extract_prefix_suffix_pairs, sequences_to_sdfa_tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import editdistance
import time
from scipy.optimize import linear_sum_assignment

from create_Seq2Seq import Encoder as LSTM_encoder
from create_Seq2Seq import Decoder as LSTM_decoder
from model_help import EventTransformer, SDFAProjector, PositionalEncoding, entropic_relevance_diff_loss
from entropic_relevance import calculate_entropic_relevance

def collate_batch_w_local_targets(batch, num_symbols, sos_token=1000, pad_token=0):
    xs, ys, _ = zip(*batch)  # we ignore the global sdfas, will recompute locally

    max_len_x = max(len(seq) for seq in xs)
    max_len_y = max(len(seq) for seq in ys)

    padded_x, mask = [], []
    padded_y_out, padded_y_in = [], []

    for seq_x, seq_y in zip(xs, ys):
        # Pad input
        padded_seq_x = seq_x + [pad_token] * (max_len_x - len(seq_x))
        padded_x.append(padded_seq_x)
        mask.append([1] * len(seq_x) + [0] * (max_len_x - len(seq_x)))

        # Pad target
        padded_seq_y = seq_y + [pad_token] * (max_len_y - len(seq_y))
        padded_y_out.append(padded_seq_y)

        shifted = [sos_token] + padded_seq_y[:-1]
        padded_y_in.append(shifted)

    x = torch.tensor(padded_x, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.bool)
    y_in = torch.tensor(padded_y_in, dtype=torch.long)
    y_out = torch.tensor(padded_y_out, dtype=torch.long)

    y_sequences = [[yi] if isinstance(yi, int) else yi for yi in ys]
    sdfa_targets = sequences_to_sdfa_tensor(y_sequences, num_symbols)#num_symbols=max(x.max().item(), y_out.max().item()))

    # Add batch dimension if needed
    if sdfa_targets.dim() == 2:
        sdfa_targets = sdfa_targets.unsqueeze(0)

    return x, mask, y_in, y_out, sdfa_targets


def collate_batch_w_targets(batch, num_symbols, sos_token=1000, pad_token=0):
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



def train_suffix_model(dataset, model, le, sequences, optimizer, er_loss, mix_lambda, device, local=False, batch_size=32, num_epochs=10):
    model = model.to(device)
    model.train()
    epoch_time = 0

    ce_losses = []
    er_losses = []
    epoch_loss_er = []
    epoch_loss_ce = []
    
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        epoch_start = time.perf_counter()  
        total_loss = 0.0

        prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
        train_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
        if local:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda b: collate_batch_w_local_targets(b, num_symbols=len(le.classes_), sos_token=le.len()+1))
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_batch_w_targets(b, num_symbols=len(le.classes_), sos_token=le.len()+1))

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
                er_losses.append(entropic_loss.item())
            else:
                loss = loss_suffix

            ce_losses.append(loss_suffix.item())            

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_end = time.perf_counter()            # ← 3. end
        epoch_time += epoch_end - epoch_start 
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
    plt.savefig(f'training_losses_suffix_{dataset}.png')
    plt.show()



    return epoch_time / num_epochs

def evaluate_suffix_model(model, le, sequences, er_loss, device, local, batch_size=32):
    model.eval()
    total_loss = 0.0
    total_dl_distance = 0.0
    eval_time = 0.0

    prefixes, suffixes, target_dfg_tensors = extract_prefix_suffix_pairs(sequences, le)
    test_dataset = EventDatasetTargets(prefixes, suffixes, target_dfg_tensors, False, pad_token=0)
    if local:
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=lambda b: collate_batch_w_local_targets(b, num_symbols=len(le.classes_), sos_token=le.len()+1))
    else:
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=lambda b: collate_batch_w_targets(b, num_symbols=len(le.classes_), sos_token=le.len()+1))


    total_entropic_relevance_pred = 0
    total_entropic_relevance_target = 0
    total_sinkhorn_distance = 0

    with torch.no_grad():
        print('Starting evaluation...')
        for dl, (x, mask, y_in, y_out, sdfa_target) in enumerate(dataloader):
            # print('New batch --- ', dl, '/', len(dataloader))
            x, mask = x.to(device), mask.to(device)
            y_in, y_out = y_in.to(device), y_out.to(device)
            sdfa_target = sdfa_target.to(device)

            if type(model) == SDFA_suffix_model_LSTM:
                embedded = model.embedding(x)
                enc_states, hidden, cell = model.encoder(embedded)
            else:
                memory = model.encoder(x, mask)


            if er_loss:
                sdfa_pred, _ = model(x, mask, y_in)
                batch_entropic_relevance_pred = 0
                batch_entropic_relevance_target = 0
                if not local:
                    for b in range(sdfa_pred.size(0)):
                        s = sdfa_pred[b]
                        L_A = s / (s.sum(dim=-1, keepdim=True) + 1e-9)
                        entropic_relevance_pred = calculate_entropic_relevance(L_A, y_out, le)
                        entropic_relevance_target = calculate_entropic_relevance(sdfa_target[b], y_out, le)
                        batch_entropic_relevance_pred += entropic_relevance_pred / len(sdfa_pred)
                        batch_entropic_relevance_target += entropic_relevance_target / len(sdfa_pred)
                    total_entropic_relevance_pred += batch_entropic_relevance_pred / len(sdfa_pred)
                    total_entropic_relevance_target += batch_entropic_relevance_target / len(sdfa_pred)
                else:
                    sdfa_pred, _ = model(x, mask, y_in)
                    batch_entropic_relevance_pred = 0
                    batch_entropic_relevance_target = 0
                    for b in range(sdfa_pred.size(0)):
                        s = sdfa_pred[b]
                        L_A = s / (s.sum(dim=-1, keepdim=True) + 1e-9)
                        entropic_relevance_pred = calculate_entropic_relevance(L_A, y_out, le)
                        entropic_relevance_target = calculate_entropic_relevance(sdfa_target[0], y_out, le)
                        batch_entropic_relevance_pred += entropic_relevance_pred / len(sdfa_pred)
                        batch_entropic_relevance_target += entropic_relevance_target / len(sdfa_pred)
                    total_entropic_relevance_pred += batch_entropic_relevance_pred / len(sdfa_pred)
                    total_entropic_relevance_target += batch_entropic_relevance_target / len(sdfa_pred)



            batch_size = x.size(0)
            max_len = y_out.size(1)
            sos_token = y_in[0, 0].item()   # your SOS value
            eoc_index = le.transform(["EOC"])[0]

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
                distance, ctime = compute_avg_damerau_levenshtein(True, pred_seq, target_seq)
                eval_time += ctime
                total_dl_distance += distance / max_len


    avg_dl_distance = total_dl_distance / len(sequences)
    print(f"Avg Damerau-Levenshtein distance on test set: {avg_dl_distance:.4f}")
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")
    print('Evaluation time for DL computation:', eval_time, 'seconds')
    print('Entropic relevance:', total_entropic_relevance_pred/len(sequences), total_entropic_relevance_target/len(sequences))
    print('Sinkhorn distance:', total_sinkhorn_distance/len(sequences))
    return total_loss/len(dataloader), avg_dl_distance, total_entropic_relevance_pred/len(sequences), total_entropic_relevance_target/len(sequences)


def compute_avg_damerau_levenshtein(use_np, suffix_pred_np, suffix_true_np):
    total_distance = 0.0


    begin_time = time.perf_counter()
    # for i in range(batch_size):
        # Remove padding (0)
    if use_np:
        pred_seq = [x for x in suffix_pred_np if x != 0]
        true_seq = [x for x in suffix_true_np if x != 0]


        distance = editdistance.eval(pred_seq, true_seq)

    total_distance += distance
    end_time = time.perf_counter()

    time_taken = end_time - begin_time
    # avg_distance = total_distance #/ batch_size
    return total_distance, time_taken


def sinkhorn_distance_batch(
    P: torch.Tensor,          # (B, H, W)
    Q: torch.Tensor,          # (B, H, W)
    reg: float = 0.01,
    num_iters: int = 500,
    eps: float = 1e-8,
    return_transport: bool = False,
) -> torch.Tensor:
    """
    Batch version of the entropic‑regularised Wasserstein‑1 distance.
    """
    # 1️⃣ sanity
    assert P.shape == Q.shape, "P and Q must have identical shapes."
    B, H, W = P.shape
    N = H * W

    # 2️⃣ flatten histograms
    a = P.reshape(B, N) + eps
    b = Q.reshape(B, N) + eps

    # 3️⃣ cost matrix (H,W grid)
    y, x = torch.meshgrid(
        torch.arange(H, device=P.device, dtype=torch.float32),
        torch.arange(W, device=P.device, dtype=torch.float32),
        indexing='ij',
    )
    coords = torch.stack([y.flatten(), x.flatten()], dim=1)   # (N, 2)
    diff = coords[:, None, :] - coords[None, :, :]           # (N, N, 2)
    C = torch.norm(diff, dim=2) + eps                        # (N, N)

    # 4️⃣ kernel
    K = torch.exp(-C / reg)                                 # (N, N)
    K_b = K.unsqueeze(0).expand(B, -1, -1)                  # (B, N, N)

    # 5️⃣ dual variables (uniform init)
    u = torch.full((B, N), 1.0 / N, dtype=P.dtype, device=P.device)
    v = torch.full((B, N), 1.0 / N, dtype=P.dtype, device=P.device)

    # 6️⃣ Sinkhorn iterations
    for _ in range(num_iters):
        u = a / (torch.bmm(K_b, v.unsqueeze(-1)).squeeze(-1) + eps)
        v = b / (torch.bmm(K_b.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps)

    # 7️⃣ transport plan
    # diag(u) @ K @ diag(v)
    diag_u = torch.diag_embed(u)           # (B, N, N)
    diag_v = torch.diag_embed(v)           # (B, N, N)
    T = torch.bmm(diag_u, torch.bmm(K_b, diag_v))  # (B, N, N)

    # 8️⃣ distance
    dist = (C[None, :, :] * T).sum(dim=(1, 2))   # (B,)

    if return_transport:
        return dist, T
    return dist


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

    