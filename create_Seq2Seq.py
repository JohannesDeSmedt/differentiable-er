"""
This module contains classes to build Seq2Seq models, including:
trace-based Seq2Seq model without attention (class: Seq2Seq_trace) and with 
attention (class: Seq2Seq_trace_attn) 
integrated Seq2Seq model without attention (class: Seq2Seq_cat) and with 
attention  (class: Seq2Seq_cat_attn) 

Classes:

    Encoder
    Decoder
    Decoder_attn
    Decoder_cat_attn
    Seq2Seq_trace
    Seq2Seq_trace_attn
    Seq2Seq_cat
    Seq2Seq_cat_attn
    
    
"""
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """

        Parameters
        ----------
        input_size: int
            Number of input features
        hidden_size: int
            Number of hidden units in LSTM
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.

        """

        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
                
    def forward(self, prefix):
        """

        Parameters
        ----------
        prefix: tensor
            Trace prefix or log prefix
            shape: (batch_size, prefix_len, num_features)

        Returns
        -------
        enc_states: tensor
            shape: (batch_size, prefix_len, hidden_size)
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)

        """
        
        enc_states, (hidden, cell) = self.lstm(prefix)

        return enc_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """

        Parameters
        ----------
        input_size: int
            Number of input features
        hidden_size: int
            Number of hidden units in LSTM.
        output_size: int
            Number of features for output.
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.

        """        
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size , 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, suffix_vector, hidden, cell):

        """
        Parameters
        ----------
        suffix_vector: tensor
            Feature vector representing an event in the suffix
            shape: (batch_size, num_features)
        hidden: tensor
            Hidden state
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            Cell state
            shape: (num_layers, batch_size, hidden_size)

        Returns
        -------
        prediction: tensor
            shape: (batch_size, output_size)
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)
        """
         
        suffix_vector = suffix_vector.unsqueeze(1) # shape: (batch_size, 1, num_features) 
        
        outputs, (hidden, cell) = self.lstm(suffix_vector, (hidden, cell)) # shape: (batch_size, 1, hidden_size)
        
        prediction = self.fc(outputs) # shape: (batch_size, 1, output_size)
        prediction = prediction.squeeze(1) # shape: (batch_size, output_size)

        return prediction, hidden, cell

class Decoder_attn(nn.Module):
    def __init__(self, input_size, enc_hidden_size, dec_hidden_size,
                 output_size, num_layers, dropout):
        """
        Parameters
        ----------
        input_size: int
            Number of input features
        enc_hidden_size: int
            Number of hidden units in the encoder LSTM.
        dec_hidden_size: int
            Number of hidden units in the decoder LSTM.
        output_size: int
            Number of features for output.
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.
        """        
        super(Decoder_attn, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=dec_hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)     
        self.proj_trace = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.proj_concate = nn.Linear(dec_hidden_size+enc_hidden_size, dec_hidden_size)
        self.fc = nn.Linear(dec_hidden_size, output_size)

    def attn(self, enc_states, top_hidden, mask):
        """
        Parameters
        ----------
        enc_states: tensor
            shape: (batch_size, prefix_len, enc_hidden_size)
        top_hidden: tensor
            shape: (batch_size, 1, dec_hidden_size)
        mask: tensor
            shape: (batch_size, prefix_len)
            True indicates positions to mask (padding).

        Returns
        -------
        attn_scores: tensor
            shape: (batch_size, prefix_len)
        context: tensor
            shape: (batch_size, enc_hidden_size)
        
        """

        attn_energies = self.proj_trace(enc_states) # shape: (batch_size, prefix_len, dec_hidden_size)
        attn_energies = attn_energies.transpose(1, 2) # shape: (batch_size, dec_hidden_size, prefix_len)

        attn_scores = torch.bmm(top_hidden, attn_energies) # shape: (batch_size, 1, prefix_len)

        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_scores = F.softmax(attn_scores, dim=-1) 

        context = torch.bmm(attn_scores, enc_states) # shape: (batch_size, 1, enc_hidden_size)

        attn_scores = attn_scores.squeeze(1) # shape: (batch_size, prefix_len)

        context = context.squeeze(1) # shape: (batch_size, enc_hidden_size)

        return attn_scores, context 
          
    def forward(self, suffix_vector, hidden, cell, enc_states, mask):
        """
        Parameters
        ----------
        suffix_vector: tensor
            shape: (batch_size, num_features)
        hidden: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        enc_states: tensor
            shape: (batch_size, prefix_len, enc_hidden_size)
        mask: tensor
            shape: (batch_size, prefix_len)
            True indicates positions to mask (padding).

        Returns
        -------
        prediction: tensor
            shape: (batch_size, output_size)
        hidden: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        attn_vector: tensor
            shape: (batch_size, dec_hidden_size)
        attn_scores: tensor
            shape: (batch_size, prefix_len)
        """

        suffix_vector = suffix_vector.unsqueeze(1) # shape: (batch_size, 1, num_features) 
        
        outputs, (hidden, cell) = self.lstm(suffix_vector, (hidden, cell)) 
        # outputs shape: (batch_size, 1, dec_hidden_size)
        # hidden shape: (num_layers, batch_size, dec_hidden_size)
        # cell shape: (num_layers, batch_size, dec_hidden_size)

        top_hidden =  hidden[-1].unsqueeze(1) # shape: (batch_size, 1, dec_hidden_size)

        attn_scores, context = self.attn(enc_states, top_hidden, mask)
        # attn_scores shape: (batch_size, prefix_len)
        # context shape: (batch_size, enc_hidden_size)

        top_hidden = top_hidden.squeeze(1) # shape: (batch_size, dec_hidden_size)

        attn_vector = torch.cat((context, top_hidden), dim=-1)  # shape: (batch_size, enc_hidden_size + dec_hidden_size)
        attn_vector = self.proj_concate(attn_vector)  # shape: (batch_size, dec_hidden_size)
        attn_vector = F.tanh(attn_vector)

        prediction = self.fc(attn_vector) # shape: (batch_size, output_size)
    
        return prediction, hidden, cell, attn_vector, attn_scores

class Decoder_cat_attn(nn.Module):
    def __init__(self, input_size, enc_hidden_size, dec_hidden_size,
                 output_size, num_layers, dropout):
        """
        Parameters
        ----------
        input_size: int
            Number of features.
        enc_hidden_size: int
            Number of hidden units in the encoder LSTM.
        dec_hidden_size: int
            Number of hidden units in the decoder LSTM.
        output_size: int
            Number of features for output.
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.
        """        
        super(Decoder_cat_attn, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=dec_hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)     
        self.proj_trace = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.proj_log = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.proj_concate = nn.Linear(dec_hidden_size+enc_hidden_size+enc_hidden_size, 
                                dec_hidden_size)
        self.fc = nn.Linear(dec_hidden_size, output_size)
    
    def attn(self, enc_states, top_hidden, mask, prefix_type):
        """
        Parameters
        ----------
        enc_states: tensor
            shape: (batch_size, prefix_len, enc_hidden_size)
        top_hidden: tensor
            shape: (batch_size, 1, dec_hidden_size)
        mask: tensor
            Masked position is true
            shape: (batch_size, prefix_len)
        prefix_type: string
            "trace" or "log"

        Returns
        -------
        attn_scores: tensor
            shape: (batch_size, prefix_len)
        context: tensor
            shape: (batch_size, enc_hidden_size)
        
        """

        if prefix_type == "trace":
            attn_energies = self.proj_trace(enc_states) # shape: (batch_size, prefix_len, dec_hidden_size)
        elif prefix_type == "log":
            attn_energies = self.proj_log(enc_states) # shape: (batch_size, prefix_len, dec_hidden_size)
        else:
            raise ValueError(f"Unknown prefix_type: {prefix_type}")
        
        attn_energies = attn_energies.transpose(1, 2) # shape: (batch_size, dec_hidden_size, prefix_len)

        attn_scores = torch.bmm(top_hidden, attn_energies) # shape: (batch_size, 1, prefix_len)

        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_scores = F.softmax(attn_scores, dim=-1) 

        context = torch.bmm(attn_scores, enc_states) # shape: (batch_size, 1, enc_hidden_size)

        attn_scores = attn_scores.squeeze(1) # shape: (batch_size, prefix_len)

        context = context.squeeze(1) # shape: (batch_size, enc_hidden_size)

        return attn_scores, context 
             
    def forward(self, suffix_vector, hidden, cell, trace_enc_states, log_enc_states, 
                trace_mask, log_mask):
        """
        Parameters
        ----------
        suffix_vector: tensor
            shape: (batch_size, num_features)
        hidden: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
            Combined last hidden states from two encoders
        cell: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
            Combined last cell states from two encoders
        trace_enc_states: tensor
            shape: (batch_size, prefix_len, enc_hidden_size)
        log_enc_states: tensor
            shape: (batch_size, prefix_len, enc_hidden_size)
        trace_mask: tensor
            shape: (batch_size, prefix_len)
            True indicates positions in trace prefix to mask (padding).
        log_mask: tensor
            shape: (batch_size, prefix_len)
            True indicates positions in log prefix to mask (padding).

        Returns
        -------
        prediction: tensor
            shape: (batch_size, output_size)
        hidden: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, dec_hidden_size)
        attn_vector: tensor
            shape: (batch_size, dec_hidden_size)
        trace_attn_scores: tensor
            shape: (batch_size, prefix_len)
        log_attn_scores: tensor
            shape: (batch_size, prefix_len)
        """

        suffix_vector = suffix_vector.unsqueeze(1) # shape: (batch_size, 1, num_features) 
        
        outputs, (hidden, cell) = self.lstm(suffix_vector, (hidden, cell)) 
        # outputs shape: (batch_size, 1, dec_hidden_size)
        # hidden shape: (num_layers, batch_size, dec_hidden_size)
        # cell shape: (num_layers, batch_size, dec_hidden_size)

        top_hidden =  hidden[-1].unsqueeze(1) # shape: (batch_size, 1, dec_hidden_size)

        trace_attn_scores, trace_context = self.attn(trace_enc_states, top_hidden, trace_mask, "trace")
        log_attn_scores, log_context = self.attn(log_enc_states, top_hidden, log_mask, "log")
        # attn_scores shape: (batch_size, prefix_len)
        # context shape: (batch_size, enc_hidden_size)

        top_hidden = top_hidden.squeeze(1) # shape: (batch_size, dec_hidden_size)

        attn_vector = torch.cat((trace_context, log_context, top_hidden), dim=-1)  # shape: (batch_size, enc_hidden_size+enc_hidden_size + dec_hidden_size)
        attn_vector = self.proj_concate(attn_vector)  # shape: (batch_size, dec_hidden_size)
        attn_vector = F.tanh(attn_vector)

        prediction = self.fc(attn_vector) # shape (batch_size, output_size)
    
        return prediction, hidden, cell, attn_vector, trace_attn_scores, log_attn_scores


class Seq2Seq_trace(nn.Module): 
    """
    One encoder, two decoders

    """            
    def __init__(self, num_act, encoder, act_decoder, time_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        encoder: object
            An instance of Class Encoder used to encode log/trace prefix.
        act_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_trace, self).__init__()
        self.num_act = num_act
        self.encoder = encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, 
                trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
            will not be used by the function. keep this parameter to 
            make training function more general
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = trace_prefix_tensor.shape[0]

        _, hidden, cell = self.encoder(trace_prefix_tensor)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        # -- for activity decoder --
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0 # shape: (batch_size, suffix_len, num_act)

        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act] # shape: (batch_size, num_act)

        act_hidden, act_cell = hidden, cell

        for t in range(act_suffix_len):

            act_prediction, act_hidden, act_cell = self.act_decoder(x_act, act_hidden, act_cell)
            # act_prediction shape: (batch_size, num_act)

            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1) # shape: (batch_size)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act) # shape: (batch_size, num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float() # shape: (batch_size, num_act)

        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1) # shape: (batch_size, suffix_length, 1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1] # shape: (batch_size)
        x_time = x_time.unsqueeze(-1) # shape: (batch_size, 1)

        time_hidden, time_cell = hidden, cell

        for t in range(time_suffix_len):

            time_prediction, time_hidden, time_cell = self.time_decoder(x_time, time_hidden, time_cell)
            # time_prediction shape: (batch_size, 1)

            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction # shape: (batch_size, 1)

        return act_predictions, time_predictions

class Seq2Seq_trace_attn(nn.Module): 
    """
    One encoder, two decoders

    """            
    def __init__(self, num_act, encoder, act_decoder, time_decoder, input_feed=False):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        encoder: object
            An instance of Class Encoder used to encode log/trace prefix.
        act_dncoder: object
            An instance of Class Decoder_attn used to generate activity label suffix prediction.
        time_dncoder: object
            An instance of Class Decoder_attn used to generate timestamp suffix prediction.
        input_feed: boolean
            Indicates whether to apply input feeding or not.
        """
        super(Seq2Seq_trace_attn, self).__init__()
        self.num_act = num_act
        self.encoder = encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder
        self.input_feed = input_feed

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, 
                trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
            will not be used by the function. keep this parameter to 
            make training function more general
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = trace_prefix_tensor.shape[0]
        trace_prefix_len = trace_prefix_tensor.shape[1]

        mask = (trace_prefix_tensor == -10000.00).any(dim=-1) # shape: (batch_size, prefix_len)
        mask.bool()
        if mask.all(dim=1).any():
            raise ValueError("At least one sample has all positions masked, causing invalid attention.")

        enc_states, hidden, cell = self.encoder(trace_prefix_tensor)
        # enc_states shape: (batch_size, prefix_len, enc_hidden_size)
        # hidden shape: (num_layers, batch_size, enc_hidden_size)
        # cell shape: (num_layers, batch_size, enc_hidden_size)

        # -- for activity decoder --
        

        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act) # shape: (batch_size, suffix_len, num_act)
        trace_act_suffix_tensor[:, :, 0] = 0 

        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act] # shape: (batch_size, num_act)        

        act_hidden, act_cell = hidden.clone(), cell.clone()

        # input feeding
        if self.input_feed:
            x_act = torch.cat((x_act, act_hidden[-1]), dim=-1) # shape: (batch_size, num_act+dec_hidden_size)

        for t in range(act_suffix_len):

            act_prediction, act_hidden, act_cell, act_attn_vector, act_attn_score = \
                self.act_decoder(x_act, act_hidden, act_cell, enc_states, mask)

            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1) # shape: (batch_size)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act) # shape: (batch_size, num_act)
            best_guess[:, 0] = 0 

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess

            # input feeding
            if self.input_feed:
                x_act = torch.cat((x_act, act_attn_vector), dim=-1) # shape: (batch_size, num_act+dec_hidden_size)
            
            x_act = x_act.float()
            
        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1] # shape: (batch_size)
        x_time = x_time.unsqueeze(-1) # shape: (batch_size, 1)

        time_hidden, time_cell = hidden.clone(), cell.clone()

        # input feeding
        if self.input_feed:
            x_time = torch.cat((x_time, time_hidden[-1]), dim=-1) # shape: (batch_size, 1+dec_hidden_size)

        for t in range(time_suffix_len):

            time_prediction, time_hidden, time_cell, time_attn_vector, time_attn_score = \
                self.time_decoder(x_time, time_hidden, time_cell, enc_states, mask)

            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

            # input feeding
            if self.input_feed:
                x_time = torch.cat((x_time, time_attn_vector), dim=-1) # shape: (batch_size, num_act+dec_hidden_size)

        return act_predictions, time_predictions
        
class Seq2Seq_cat(nn.Module): 
    """
    Two encoders, two decoders

    """           
    def __init__(self, num_act, log_encoder, trace_encoder, act_cat_decoder, time_cat_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_cat, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_cat_decoder
        self.time_decoder = time_cat_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = log_prefix_tensor.shape[0]

        _, log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        # log_hidden shape: (num_layers, batch_size, enc_hidden_size)
        # log_cell shape: (num_layers, batch_size, enc_hidden_size)
        _, trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)
        # trace_hidden shape: (num_layers, batch_size, enc_hidden_size)
        # trace_cell shape: (num_layers, batch_size, enc_hidden_size)

        hidden = torch.cat((log_hidden, trace_hidden), -1) # shape: (num_layers, batch_size, enc_hidden_size * 2)
        cell = torch.cat((log_cell, trace_cell), -1) # shape: (num_layers, batch_size, enc_hidden_size * 2)

        # -- for activity decoder --
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        act_hidden, act_cell = hidden, cell

        for t in range(act_suffix_len):

            act_prediction, act_hidden, act_cell = self.act_decoder(x_act, act_hidden, act_cell)
            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()

        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]
        x_time = x_time.unsqueeze(-1)

        time_hidden, time_cell = hidden, cell

        for t in range(time_suffix_len):

            time_prediction, time_hidden, time_cell = self.time_decoder(x_time, time_hidden, time_cell)
            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

        return act_predictions, time_predictions
    
class Seq2Seq_cat_attn(nn.Module): 
    """
    Two encoders, two decoders

    """           
    def __init__(self, 
                 num_act, 
                 log_encoder, trace_encoder, act_cat_decoder, time_cat_decoder, 
                 input_feed=False):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder_cat_attn used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder_cat_attn used to generate timestamp suffix prediction.
        input_feed: boolean
            Indicates whether to apply input feeding or not.
        """
        super(Seq2Seq_cat_attn, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_cat_decoder
        self.time_decoder = time_cat_decoder
        self.input_feed = input_feed

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, 
                trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
            input of Seq2Seq model
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
            input of Seq2Seq model
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = log_prefix_tensor.shape[0]
        trace_prefix_len = trace_prefix_tensor.shape[1]
        log_prefix_len = log_prefix_tensor.shape[1]

        trace_mask = (trace_prefix_tensor == -10000.00).any(dim=-1) # shape: (batch_size, prefix_len)
        trace_mask.bool()
        if trace_mask.all(dim=1).any():
            raise ValueError("At least one sample has all positions masked, causing invalid attention.")
        
        log_mask = (log_prefix_tensor == -10000.00).any(dim=-1) # shape: (batch_size, prefix_len)
        log_mask.bool()
        if log_mask.all(dim=1).any():
            raise ValueError("At least one sample has all positions masked, causing invalid attention.")

        log_enc_states, log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        trace_enc_states, trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)
        # enc_states shape: (batch_size, prefix_len, enc_hidden_size)
        # hidden shape: (num_layers, batch_size, enc_hidden_size)
        # cell shape: (num_layers, batch_size, enc_hidden_size)

        hidden = torch.cat((log_hidden, trace_hidden), -1) # shape: (num_layers, batch_size, enc_hidden_size * 2)
        cell = torch.cat((log_cell, trace_cell), -1) # shape: (num_layers, batch_size, enc_hidden_size * 2)

        # -- for activity decoder --
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0  # Ensure padding is represented by all 0s

        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        act_hidden, act_cell = hidden.clone(), cell.clone()

        # input feeding
        if self.input_feed:
            x_act = torch.cat((x_act, act_hidden[-1]), dim=-1) # shape: (batch_size, num_act+dec_hidden_size)

        for t in range(act_suffix_len):
            
            act_prediction, act_hidden, act_cell, act_attn_vector, \
                act_trace_attn_score, act_log_attn_score = \
                    self.act_decoder(x_act, act_hidden, act_cell, 
                                     trace_enc_states, log_enc_states,
                                     trace_mask, log_mask)

            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0 

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            
            # input feeding
            if self.input_feed:
                x_act = torch.cat((x_act, act_attn_vector), dim=-1) # shape: (batch_size, num_act+dec_hidden_size)
            
            x_act = x_act.float()
            
        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]

        x_time = x_time.unsqueeze(-1)

        time_hidden, time_cell = hidden.clone(), cell.clone()

        # input feeding
        if self.input_feed:
            x_time = torch.cat((x_time, time_hidden[-1]), dim=-1)

        for t in range(time_suffix_len):
            
            # pay attention to the input
            time_prediction, time_hidden, time_cell, time_attn_vector, \
                time_trace_attn_score, time_log_attn_score = \
                    self.time_decoder(x_time, time_hidden, time_cell, 
                                      trace_enc_states, log_enc_states,
                                      trace_mask, log_mask)

            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

            # input feeding
            if self.input_feed:
                x_time = torch.cat((x_time, time_attn_vector), dim=-1)

        return act_predictions, time_predictions