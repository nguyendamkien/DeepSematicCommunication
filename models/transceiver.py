# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
这是一个Transformer的网络结构
"""
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"create mask nt, perturb attention"
class MaskPerturbation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)

    def forward(self, Q, K):
        """
        Q: (batch, seq_len, d_model)
        K: (batch, seq_len, d_model)
        """
        q_proj = self.wq(Q) #q*Wq
        k_proj = self.wk(K) #k*Wk

        d_k = Q.size(-1)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(d_k)
        
        m = torch.sigmoid(scores)  # mask m_t

        return m
    
# perturbed attention weight
def perturb_attention_weight(attn, mask):
    """
    attn: (batch, heads, seq_len, seq_len)
    mask: (batch, seq_len, seq_len)
    """
    # expand mask cho multi-head
    # mask: (batch, 1, seq_len, seq_len)
    mask = mask.unsqueeze(1)

    seq_len = attn.size(-1)
    uniform = torch.ones_like(attn) / seq_len

    attn_perturbed_weight = mask * attn + (1 - mask) * uniform

    return attn_perturbed_weight

# calibrated attention weight
def calibrate_attention_weight(attn, mask):
    """
    attn: (batch, heads, seq_len, seq_len)
    mask: (batch, seq_len, seq_len)
    """
    mask = mask.unsqueeze(1)

    attn_cal = attn * torch.exp(1 - mask)

    return attn_cal

class AttentionCalibration(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)

    def forward(self, Q, attn, attn_cal):
        """
        Q: (batch, seq_len, d_model)
        attn, attn_cal: (batch, heads, seq_len, seq_len)
        """
        g = torch.sigmoid(self.gate(Q))  # (batch, seq_len, 1)
        g = g.unsqueeze(1)  # expand for heads

        attn_comb = g * attn + (1 - g) * attn_cal

        return F.softmax(attn_comb, dim=-1)
    
class CalibratedMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.mask_perturbation_model = MaskPerturbation(d_model)
        self.calibration = AttentionCalibration(d_model)

    def forward(self, query, key, value, mask=None, use_perturb=False):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query_origin = query
        key_origin = key

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # mask perturbation
        mask_perturbation = self.mask_perturbation_model(query_origin, key_origin)

        # attention gốc
        x, attn = self.attention(query, key, value, mask=mask)

        if use_perturb:
            # dùng attention bị phá
            attn_final = perturb_attention_weight(attn, mask_perturbation)
        else:
            # calibration
            attn_cal = calibrate_attention_weight(attn, mask_perturbation)
            attn_final = self.calibration(query_origin, attn, attn_cal)

        # output
        out = torch.matmul(attn_final, value)

        out = out.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)

        out = self.dense(out)

        out = self.dropout(out)

        return out, mask_perturbation
    
    def attention(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            scores = scores + (mask * -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn
    
# loss_mask = - loss_nmt(attn_perturbed) + alpha * torch.norm(1 - mask)

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


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
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

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn
    
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
    
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    """Single layer of the semantic encoder
    Input: [batch_size, seq_len, d_model]
    Output: [batch_size, seq_len, d_model]"""

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-head attention for processing input sequence
        # Input/Output: [batch_size, seq_len, d_model]
        self.mha = MultiHeadedAttention(num_heads, d_model)

        # Position-wise feed-forward network
        # Input/Output: [batch_size, seq_len, d_model]
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        """
        Input:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len] - Attention mask
        Output: [batch_size, seq_len, d_model]
        """
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x
    
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    """Single layer of the semantic decoder
    Input: [batch_size, seq_len, d_model]
    Output: [batch_size, seq_len, d_model]"""

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model,
                                             dropout=0.1)  # Masked self-attention
        self.src_mha = CalibratedMultiHeadAttention(num_heads, d_model,
                                            dropout=0.1)  # Encoder-decoder attention
        self.ffn = PositionwiseFeedForward(d_model, dff,
                                           dropout=0.1)  # Feedforward network

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask, use_perturb=False):
        """Follow Figure 1 (right) for connections."""
        """
        Inputs:
            x: [batch_size, tgt_seq_len, d_model] - Target sequence
            memory: [batch_size, src_seq_len, d_model] - Channel Decoder output
            look_ahead_mask: [batch_size, tgt_seq_len, tgt_seq_len] - Causal mask
            trg_padding_mask: [batch_size, tgt_seq_len, src_seq_len] - Cross-attention mask
        Output: ([batch_size, tgt_seq_len, d_model], mask_perturbation)
        """
        # m = memory

        # 1. Masked self-attention (target sequence attending to itself)
        attn_output = self.self_mha(x, x, x,
                                    look_ahead_mask)  # Q, K, V are all 'x'
        x = self.layernorm1(
            x + attn_output)  # Residual connection + normalization

        # 2. Cross-attention (decoder attends to encoder output)
        src_output, m_t = self.src_mha(x, memory, memory,
                                  trg_padding_mask, use_perturb=use_perturb )  # Q=x, K=V=memory
        x = self.layernorm2(x + src_output)

        # 3. Feedforward network
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)

        return x, m_t
    
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    """Complete semantic encoder
    Input: Token indices [batch_size, seq_len]
    Output: Semantic features [batch_size, seq_len, d_model]"""

    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout)
             for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        """Pass the input (and mask) through each layer in turn."""
        """
        Input:
            x: [batch_size, seq_len] - Token indices
            src_mask: [batch_size, 1, seq_len] - Attention mask
        # the input size of x is [batch_size, seq_len]
        Output: [batch_size, seq_len, d_model]
        """
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        return x
    
class Decoder(nn.Module):
    """Complete semantic decoder
    Input: Token indices [batch_size, seq_len]
    Output: Features [batch_size, seq_len, d_model]"""

    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size,
                                      d_model)  # Token embedding
        self.pos_encoding = PositionalEncoding(d_model, dropout,
                                               max_len)  # Positional encoding
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout)
             for _ in range(num_layers)])  # Stack of decoder layers
        
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask, use_perturb=False):
        """
        Inputs:
            x: [batch_size, tgt_seq_len] - Target tokens
            memory: [batch_size, src_seq_len, d_model] - Encoder output
            look_ahead_mask: [batch_size, tgt_seq_len, tgt_seq_len]
            trg_padding_mask: [batch_size, tgt_seq_len, src_seq_len]
        Output: ([batch_size, tgt_seq_len, d_model], mask_perturbation)
        """
        # Convert token indices to embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embedding
        x = self.pos_encoding(x)  # Add positional encoding

        # Pass through each decoder layer
        m_t = None
        for dec_layer in self.dec_layers:
            x, m_t = dec_layer(x, memory, look_ahead_mask, trg_padding_mask, use_perturb)

        return x, m_t  # Final decoder output and mask perturbation
    
class ChannelDecoder(nn.Module):
    """Channel decoder for converting channel-coded features back to semantic space
    Input: Channel features [batch_size, seq_len, in_features]
    Output: Semantic features [batch_size, seq_len, size1]"""

    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        """
        Input: x [batch_size, seq_len, in_features] (16)
        Output: [batch_size, seq_len, size1] (d_model)
        """
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output
    
# Model
class DeepSC(nn.Module):
    """Complete DeepSC model combining semantic and channel coding
    Input: Token indices [batch_size, seq_len]
    Output: Vocabulary logits [batch_size, seq_len, vocab_size]"""

    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(DeepSC, self).__init__()

        # Semantic encoder
        # Input: [batch_size, seq_len] -> Output: [batch_size, seq_len, d_model]
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)

        # Channel encoder
        # Input: [batch_size, seq_len, d_model] -> Output: [batch_size, seq_len, 16]
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256),
                                             # nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))

        # Channel decoder
        # Input: [batch_size, seq_len, 16] -> Output: [batch_size, seq_len, d_model]
        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        # Semantic decoder
        # Input: [batch_size, seq_len] -> Output: [batch_size, seq_len, d_model]
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        # Final output layer
        # Input: [batch_size, seq_len, d_model] -> Output: [batch_size, seq_len, trg_vocab_size]
        self.dense = nn.Linear(d_model, trg_vocab_size)



