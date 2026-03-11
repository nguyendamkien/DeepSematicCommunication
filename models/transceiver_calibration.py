import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(
                                 10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x
    
class CalibratedSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(CalibratedSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, calibration=None):
        "Implements Figure 2"
        if calibration is not None:
            # Same mask applied to all h heads.
            calibration = calibration.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, calibration=calibration)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, calibration=None, mask=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if calibration is not None:
            scores = scores * calibration
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn
    
"""
Model Deep Semantic Communication use self-celibration attention machenism
Transformer using celibration self-attention includes:
    Encoder
        1. Positional coding
        2. self-celibration attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""

class DetectionNet(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        # GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        # Linear
        self.linear = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x: [batch, seq_len, d_model]

        h, _ = self.gru(x)

        logits = self.linear(h)

        prob = self.sigmoid(logits)

        # output: [batch, seq_len]
        # xac suat token bi loi - calibration
        return prob.squeeze(-1)

class EncoderLayer(nn.Module):
    """Encoder is made up of self-calibration attn and feed forward (defined below)"""
    """Single layer of the semantic encoder
    Input: [batch_size, seq_len, d_model]
    Output: [batch_size, seq_len, d_model]"""

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.cattn = CalibratedSelfAttention(num_heads, d_model, dropout=0.1)

        # Position-wise feed-forward network
        # Input/Output: [batch_size, seq_len, d_model]
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, calibration):

        attn_output = self.cattn(x, x, x, calibration)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, 0.1)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff)
            for _ in range(num_layers)
        ])

        self.detector = DetectionNet(d_model)

    def forward(self, x):

        x = self.embedding(x)
        
        x = self.pos_encoding(x)

        C = None

        for layer in self.layers:

            if C is None:
                x = layer(x, None)
            else:
                x = layer(x, C)

            # C = error_prob or calibration
            # P = [batch, seq]
            # P_outer = [batch, seq, seq]
            # C = [batch, seq, seq]
            P = self.detector(x) 
            P_outer = torch.bmm(P.unsqueeze(2), P.unsqueeze(1))
            C = 1 - P_outer

        return x
