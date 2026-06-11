# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import json
import math
import pickle
import random
from datetime import datetime
import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchvision.models import RegNet_X_8GF_Weights
from w3lib.html import remove_tags

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nltk.download('punkt_tab')

class BleuScore:
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights
        self.smoothing = SmoothingFunction().method1  # Add smoothing function

    def compute_blue_score(self, real, predicted):
        score = []
        # ADDED: Progress bar for sentence pairs
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2,
                                       weights=(
                                           self.w1, self.w2, self.w3,
                                           self.w4),
                                       smoothing_function=self.smoothing))
        return score

# initialize the weights for the pytorch model
def initNetParams(model):
    """
    Init net parameters
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x

def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def create_masks(src, trg, padding_idx):
    # Create mask for source sequence padding
    src_mask = (src == padding_idx).unsqueeze(-2).type(
        torch.FloatTensor)  # [batch, 1, seq_len]
    # Create mask for target sequence padding
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(
        torch.FloatTensor)  # [batch, 1, seq_len]
    # Create look-ahead mask for decoder
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    # Combine padding and look-ahead masks
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    # Calculate raw loss
    loss = criterion(x, trg)
    # Create mask to ignore padding tokens
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    # Apply mask to loss
    loss *= mask
    # Return average loss over non-padding positions
    return loss.mean()

class Channels():
    def AWGN(self, Tx_sig, n_var):
        """
        Additive White Gaussian Noise channel.
        - Tx_sig: Input signal tensor
        - n_var: Noise standard deviation
        Returns:
        - Rx_sig: Received signal with noise
        - batch_snr_db: Average SNR in dB
        """
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        # SNR = signal_power / noise_power
        # Assuming signal power = 1 (normalized), noise power = n_var^2
        noise_power = n_var ** 2
        snr = 1 / noise_power
        batch_snr_db = 10 * math.log10(snr)  # Convert to dB
        return Rx_sig, batch_snr_db
    
    def Rayleigh(self, Tx_sig, n_var):
        """
        Rayleigh fading channel with equalization.
        - Tx_sig: Input signal tensor
        - n_var: Noise variance (standard deviation)
        Returns:
        - Rx_sig: Equalized received signal
        - batch_snr_db: Average SNR in dB
        """
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)  # AWGN returns (Rx_sig, snr_db)
        Rx_sig, _ = Rx_sig  # Ignore AWGN's SNR, compute post-equalization

        # Equalization
        H_inv = torch.inverse(H)
        Rx_sig = torch.matmul(Rx_sig, H_inv).view(shape)

        # Noise power after equalization: 2 * n_var^2 * ||H_inv||_F^2
        noise_power = 2 * n_var ** 2 * torch.norm(H_inv, p='fro') ** 2
        snr = 1 / noise_power
        batch_snr_db = 10 * math.log10(snr.item())
        return Rx_sig, batch_snr_db
    
    def Rician(self, Tx_sig, n_var, K=1):
        """
        Rician fading channel with equalization.
        - Tx_sig: Input signal tensor
        - n_var: Noise variance (standard deviation)
        - K: Rician factor (default=1)
        Returns:
        - Rx_sig: Equalized received signal
        - batch_snr_db: Average SNR in dB
        """
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)  # AWGN returns (Rx_sig, snr_db)
        Rx_sig, _ = Rx_sig  # Ignore AWGN's SNR, compute post-equalization

        # Equalization
        H_inv = torch.inverse(H)
        Rx_sig = torch.matmul(Rx_sig, H_inv).view(shape)

        # Noise power after equalization
        noise_power = 2 * n_var ** 2 * torch.norm(H_inv, p='fro') ** 2
        snr = 1 / noise_power
        batch_snr_db = 10 * math.log10(snr.item())
        return Rx_sig, batch_snr_db
    
    def TimeVaryingRician(self, Tx_sig, n_var, M_options=[3, 5, 6, 10], K=1,
                          csi_lag=1):
        """
        Time-varying Rician channel with block-wise fading and lagged equalization.
        Simulates a fast-moving receiver (e.g., 50–150 km/h) where the channel changes
        every block (~2 ms), and equalization uses an outdated channel estimate (lagged
        by csi_lag blocks, default=1 for ~2 ms lag, realistic for 5G with frequent pilots).
        - Tx_sig: Input signal tensor [batch_size, sequence_length, feature_dim]
        - n_var: Noise standard deviation
        - M_options: List of possible block sizes (e.g., [3, 5, 6, 10])
        - K: Rician factor (default=1 for urban mobile scenario)
        - csi_lag: Number of blocks to lag channel estimate (default=1)
        Returns:
        - Rx_sig_equalized: Equalized received signal
        - batch_snr_db: Average SNR in dB
        """
        batch_size, sequence_length, feature_dim = Tx_sig.shape
        assert feature_dim % 2 == 0, f"feature_dim {feature_dim} must be even for complex symbols"
        complex_length = sequence_length * (feature_dim // 2)

        # Select valid block size M
        valid_M_options = [m for m in M_options if complex_length % m == 0]
        if not valid_M_options:
            raise ValueError(
                f"No M in {M_options} divides complex_length {complex_length}")
        M = random.choice(valid_M_options)
        number_of_blocks = complex_length // M

        # Reshape input signal for block-wise processing
        Tx_sig_reshaped = Tx_sig.view(batch_size, number_of_blocks, M, 2)

        # Generate block-wise fading coefficients
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real_blocks = torch.normal(mean, std,
                                     size=[batch_size, number_of_blocks, 1]).to(
            device)
        H_imag_blocks = torch.normal(mean, std,
                                     size=[batch_size, number_of_blocks, 1]).to(
            device)
        H_blocks = torch.zeros(batch_size, number_of_blocks, 2, 2,
                               device=device)
        H_blocks[:, :, 0, 0] = H_real_blocks[:, :, 0]
        H_blocks[:, :, 0, 1] = -H_imag_blocks[:, :, 0]
        H_blocks[:, :, 1, 0] = H_imag_blocks[:, :, 0]
        H_blocks[:, :, 1, 1] = H_real_blocks[:, :, 0]

        # Apply block-wise fading
        Tx_sig_after_channel_reshaped = torch.matmul(Tx_sig_reshaped, H_blocks)
        Tx_sig_after_channel = Tx_sig_after_channel_reshaped.view(batch_size,
                                                                  complex_length,
                                                                  2)
        Rx_sig, _ = self.AWGN(Tx_sig_after_channel, n_var)  # Ignore AWGN SNR

        # Equalization with lagged channel estimate
        Rx_sig_reshaped = Rx_sig.view(batch_size, number_of_blocks, M, 2)
        H_blocks_lagged = torch.zeros_like(H_blocks)
        # Initial H (e.g., from pilot before transmission)
        H_initial = torch.zeros(batch_size, 2, 2, device=device)
        H_initial[:, 0, 0] = torch.normal(mean, std, size=[batch_size]).to(
            device)
        H_initial[:, 0, 1] = -torch.normal(mean, std, size=[batch_size]).to(
            device)
        H_initial[:, 1, 0] = H_initial[:, 0, 1].clone()
        H_initial[:, 1, 1] = H_initial[:, 0, 0].clone()
        # Assign lagged H: block 1 uses H_initial, block 2 uses H_blocks[:,0], etc.
        for i in range(number_of_blocks):
            if i < csi_lag:
                H_blocks_lagged[:, i, :, :] = H_initial
            else:
                H_blocks_lagged[:, i, :, :] = H_blocks[:, i - csi_lag, :, :]
        H_inv_blocks = torch.inverse(H_blocks_lagged)
        Rx_sig_equalized_reshaped = torch.matmul(Rx_sig_reshaped, H_inv_blocks)
        Rx_sig_equalized = Rx_sig_equalized_reshaped.view(batch_size,
                                                          sequence_length,
                                                          feature_dim)

        # Noise power using mean norm for realistic SNR
        fro_norm_squared = torch.norm(H_inv_blocks, p='fro', dim=[2, 3]) ** 2
        noise_power = 2 * n_var ** 2 * torch.mean(fro_norm_squared, dim=1)
        snr = 1 / noise_power
        batch_snr = torch.mean(snr).item()
        batch_snr_db = 10 * math.log10(batch_snr)

        return Rx_sig_equalized, batch_snr_db
    
def train_step(model, src, trg, n_var, pad, opt, criterion, channel):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channels()

    # remove former gradient
    opt.zero_grad()

    # mask for transformer
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    # encoder + channel encoder
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    # Channel transmission
    if channel == 'AWGN':
        Rx_sig, snr = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig, snr = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig, snr = channels.Rician(Tx_sig, n_var)
    elif channel == 'TimeVaryingRician':
        Rx_sig, snr = channels.TimeVaryingRician(Tx_sig, n_var)

    # channel decoder + decoder
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask,
                               src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)

    # calculate loss
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1), pad, criterion)

    # backprop + update
    loss.backward()
    opt.step()

    return loss.item(), snr


def val_step(model, src, trg, n_var, pad, criterion, channel, seq_to_text):
    model.eval()
    with torch.no_grad():
        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]
        channels = Channels()

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig, snr = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig, snr = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig, snr = channels.Rician(Tx_sig, n_var)
        elif channel == 'TimeVaryingRician':
            Rx_sig, snr = channels.TimeVaryingRician(Tx_sig, n_var)

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask,
                                   src_mask)
        pred = model.dense(dec_output)
        ntokens = pred.size(-1)
        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1), pad, criterion)

    return loss.item(), snr

class SeqtoText:
    def __init__(self, vocab_dictionary, end_idx):
        self.reverse_word_map = dict(
            zip(vocab_dictionary.values(), vocab_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        # Print the raw list of indices
        # print(f"Raw Indices: {list_of_indices}")
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                words.append("<END>")  # Append <END> explicitly
                # print("Encountered <END>, stopping translation.")
                break
            else:
                word = self.reverse_word_map.get(idx,
                                                 "<UNK>")  # Use <UNK> for unknown indices
                words.append(word)
            # Print each index and corresponding word
            # print(f"Index: {idx} -> Word: {word}")
        words = ' '.join(words)
        return (words)
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol,
                  channel, device):
    """
    Greedy decoder for inference/validation, supporting 3GPP channel with fixed distance and SNR.
    """
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(
        device)  # [batch, 1, seq_len]

    # Encoder and channel encoding (same as train_step)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(
        channel_enc_output)  # Assuming power_normalize is defined

    # Channel simulation
    if channel == 'AWGN':
        Rx_sig, snr = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig, snr = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig, snr = channels.Rician(Tx_sig, n_var)
    elif channel == 'TimeVaryingRician':
        Rx_sig, snr = channels.TimeVaryingRician(Tx_sig, n_var)
    else:
        raise ValueError(
            "Please choose from AWGN, Rayleigh, Rician, TimeVaryingRician, or 3GPP")

    # Decode received signal
    memory = model.channel_decoder(Rx_sig)

    # Initialize output sequence with start symbol
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    # Autoregressive generation
    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(
            torch.FloatTensor).to(device)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(
            torch.FloatTensor).to(device)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        # Generate next token
        dec_output = model.decoder(outputs, memory, combined_mask, src_mask)
        pred = model.dense(dec_output)
        # Select most probable token for next position
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        _, next_word = torch.max(prob, dim=-1)
        # Append new token to output sequence
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs, snr

def save_evaluation_scores(args, SNR, bleu_score, similarity_score, method,
                           bleu_ngram):
    """
    Save evaluation scores to CSV files in the evaluation_result/{method}-{channel} directory.

    Args:
        args: Arguments containing channel type and other model parameters
        SNR: List of SNR values used in evaluation
        bleu_score: Array of BLEU scores corresponding to SNR values
        similarity_score: Array of similarity scores corresponding to SNR values
        method: String indicating the method tested (e.g., 'deepsc', 'huffman + rs')
        bleu_ngram: Integer specifying the n-gram level of the BLEU score (e.g., 1, 2, 3, 4)
    Returns:
        str: Path to the results directory
    """
    # Create base results directory with method included
    # results_dir = f"evaluation_result/{method}-{args.channel}"
    # os.makedirs(results_dir, exist_ok=True)

    results_dir = f"/kaggle/working/{method}-{args.channel}"
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamp for files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'SNR': SNR,
        f'BLEU-{bleu_ngram}_Score': bleu_score,
        # Label BLEU column with n-gram level
        'Similarity_Score': similarity_score
    })

    # Save detailed results to CSV with method, n-gram, and timestamp
    scores_path = os.path.join(results_dir,
                               f'scores_{method}_bleu-{bleu_ngram}_{timestamp}.csv')
    results_df.to_csv(scores_path, index=False)

    # Save model configuration with additional method and n-gram info
    config = {
        'method': method,
        'bleu_ngram': bleu_ngram,
        'channel_type': args.channel,
        'max_length': args.MAX_LENGTH,
        'min_length': args.MIN_LENGTH,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dff': args.dff,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'average_bleu': float(np.mean(bleu_score)),
        'average_similarity': float(np.mean(similarity_score))
    }

    # Save configuration as JSON with method, n-gram, and timestamp
    config_path = os.path.join(results_dir,
                               f'config_{method}_bleu-{bleu_ngram}_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nResults saved to: {results_dir}")
    print(f"Scores file: scores_{method}_bleu-{bleu_ngram}_{timestamp}.csv")
    print(f"Config file: config_{method}_bleu-{bleu_ngram}_{timestamp}.json")
    return results_dir


def debug_greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol,
                        channel, seq_to_text=None):
    """Enhanced greedy decode with symbol rate, channel analysis, and signal integrity checks"""
    channels = Channels()
    device = src.device

    def print_step(step_name, tensor, show_values=True, limit=5):
        print(f"\n{'=' * 10} {step_name} {'=' * 10}")
        print(f"Shape: {list(tensor.shape)}")
        print(f"Dtype: {tensor.dtype}")
        if show_values and tensor.numel() < 200:
            print(f"Values:\n{tensor.cpu().detach().numpy()[:limit]}")
            if seq_to_text and tensor.dim() == 2:
                try:
                    text = seq_to_text.sequence_to_text(
                        tensor[0].cpu().numpy().tolist())
                    print(f"First sequence (text): {text}")
                except Exception as e:
                    print(f"[Warning] Could not convert to text: {e}")
        # Signal integrity check (only for floating-point or complex tensors)
        if tensor.dtype in [torch.float16, torch.float32, torch.float64,
                            torch.complex64, torch.complex128]:
            print(
                f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
        else:
            print(
                "Mean and Std skipped: Tensor dtype is not floating-point or complex")

    # Input Processing
    print_step("Input Sequence", src)

    # Source Mask
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)
    print_step("Source Mask", src_mask)

    # Encoder Pass
    enc_output = model.encoder(src.to(device), src_mask.to(device))
    print_step("Encoder Output", enc_output)

    # Channel Encoding with Modulation Analysis
    channel_enc_output = model.channel_encoder(enc_output.to(device))
    print_step("Channel Encoder Output", channel_enc_output)

    # Symbol Rate Estimation
    print(f"\n{'=' * 10} Symbol Rate Estimation {'=' * 10}")
    batch_size, seq_len, feat_dim = channel_enc_output.shape
    symbols_per_token = feat_dim // 2  # Assuming 2 dims per complex symbol
    total_symbols = seq_len * symbols_per_token
    print(f"Tokens: {seq_len}, Features per token: {feat_dim}")
    print(
        f"Symbols per token: {symbols_per_token}, Total symbols: {total_symbols}")
    assumed_time = 0.045  # Hypothetical duration (45 ms, adjustable)
    Rs_est = total_symbols / assumed_time
    print(
        f"Assumed transmission time: {assumed_time} s, Estimated Rs: {Rs_est:.2f} symbols/s")

    # Modulation Analysis (Pre-Channel)
    try:
        real_imag = channel_enc_output.view(batch_size, seq_len,
                                            symbols_per_token, 2)
        symbols = real_imag[..., 0] + 1j * real_imag[..., 1]
        avg_magnitude_pre = torch.mean(torch.abs(symbols))
        print(f"Pre-channel avg symbol magnitude: {avg_magnitude_pre:.4f}")
    except Exception as e:
        print(f"[Warning] Pre-channel modulation analysis failed: {e}")

    # Power Normalization
    Tx_sig = PowerNormalize(channel_enc_output)
    print_step("Transmitted Signal", Tx_sig)

    # Channel Simulation with Detailed SNR
    if channel == 'AWGN':
        Rx_sig, snr = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig, snr = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig, snr = channels.Rician(Tx_sig, n_var)
    elif channel == 'TimeVaryingRician':
        Rx_sig, snr = channels.TimeVaryingRician(Tx_sig, n_var)
    else:
        raise ValueError(
            "Please choose from AWGN, Rayleigh, Rician, or TimeVaryingRician")
    print_step("Received Signal", Rx_sig)
    print(f"Noise Variance: {n_var}, SNR: {snr:.2f} dB")

    # Post-Channel Modulation Analysis
    print(f"\n{'=' * 10} Post-Channel Analysis {'=' * 10}")
    try:
        real_imag_post = Rx_sig.view(batch_size, seq_len, symbols_per_token, 2)
        symbols_post = real_imag_post[..., 0] + 1j * real_imag_post[..., 1]
        avg_magnitude_post = torch.mean(torch.abs(symbols_post))
        print(f"Post-channel avg symbol magnitude: {avg_magnitude_post:.4f}")
    except Exception as e:
        print(f"[Warning] Post-channel modulation analysis failed: {e}")

    # Channel Decoding
    memory = model.channel_decoder(Rx_sig.to(device))
    print_step("Channel Decoder Output", memory)

    # Initialize Output
    outputs = torch.full((src.size(0), 1), start_symbol, dtype=src.dtype,
                         device=device)
    print_step("Initial Output", outputs)

    # Autoregressive Decoding
    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).float()
        look_ahead_mask = subsequent_mask(outputs.size(1)).float()
        combined_mask = torch.max(trg_mask.to(device),
                                  look_ahead_mask.to(device))
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        _, next_word = torch.max(pred[:, -1:, :], dim=-1)
        outputs = torch.cat([outputs, next_word.to(device)], dim=1)
        if next_word.item() == seq_to_text.end_idx:
            break

    # Final Output
    print("\n=== Final Output (First Sequence) ===")
    if seq_to_text:
        text = seq_to_text.sequence_to_text(outputs[0].cpu().numpy().tolist())
        print(f"Sequence 1: {text}")

    return outputs
    
def list_checkpoints(checkpoint_dir, device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"), fields_to_print=None):
    """
    Lists all checkpoints in the directory and returns their epochs, losses, timestamps, and fields.

    Args:
        checkpoint_dir (str): Directory containing .pth checkpoint files.
        device (torch.device): Device to map the checkpoint to.
        fields_to_print (list, optional): List of field names to print for each checkpoint.
                                         If None, prints all fields.

    Returns:
        tuple: (epochs, train_losses, val_losses, paths, timestamps, fields) - Lists of epochs,
               training losses, validation losses, file paths, timestamps, and checkpoint fields.
    """
    # If checkpoint directory doesn't exist, create it so callers won't fail
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoint directory '{checkpoint_dir}' did not exist; created it.")
        except Exception as e:
            print(f"Unable to create checkpoint directory '{checkpoint_dir}': {e}")
            return [], [], [], [], [], []
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found.")
        return [], [], [], [], [], []

    epochs = []
    train_losses = []
    val_losses = []
    paths = []
    timestamps = []
    fields_list = []

    for ckpt in sorted(checkpoints):  # Sort for order
        path = os.path.join(checkpoint_dir, ckpt)
        try:
            checkpoint = torch.load(path, map_location=device,
                                    weights_only=False)
            epoch = checkpoint.get('epoch', None)
            train_loss = checkpoint.get('train_loss', None)
            val_loss = checkpoint.get('loss', None)

            # Extract timestamp from filename
            timestamp_str = ckpt.split('checkpoint_')[-1].replace('.pth', '')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')

            # Get all fields in the checkpoint
            fields = list(checkpoint.keys())
            fields_list.append(fields)

            if epoch is not None and train_loss is not None:
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(
                    val_loss if val_loss is not None else float('nan'))
                paths.append(path)
                timestamps.append(timestamp)

                # Determine fields to print
                print_fields = fields if fields_to_print is None else [f for f
                                                                       in
                                                                       fields_to_print
                                                                       if
                                                                       f in fields]
                if not print_fields:
                    print_fields = [
                        'N/A']  # In case no requested fields are found

                print(f"{ckpt}: Epoch {epoch}, Train Loss {train_loss:.5f}, "
                      f"Val Loss {val_loss if val_loss is not None else 'N/A':.5f}, "
                    #   f"Timestamp {timestamp}{mi_bits_str}")
                      f"Timestamp {timestamp}")
                print(f"  Fields: {', '.join(print_fields)}")
        except Exception as e:
            print(f"Error loading {ckpt}: {e}")

    if not epochs:
        print("No valid checkpoint data found.")

    return epochs, train_losses, val_losses, paths, timestamps, fields_list

def load_checkpoint(checkpoint_dir, mode='latest'):
    """
    Loads the latest or best checkpoint based on the mode.
    - 'latest': Loads the most recent checkpoint based on timestamp.
    - 'best': Loads the checkpoint with the lowest validation loss.
    """
    # List all .pth files in the checkpoint directory
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None

    if mode == 'latest':
        # Sort by timestamp in filename to find the latest checkpoint
        latest_ckpt = max(checkpoints, key=lambda x: datetime.strptime(
            x.split('_')[1] + '_' + x.split('_')[2].split('.')[0],
            '%Y-%m-%d_%H-%M-%S'))
        path = os.path.join(checkpoint_dir, latest_ckpt)
    elif mode == 'best':
        # Find checkpoint with the lowest loss
        losses = []
        for ckpt in checkpoints:
            path = os.path.join(checkpoint_dir, ckpt)
            try:
                # Load checkpoint with weights_only=False to include non-tensor fields
                checkpoint = torch.load(path, map_location=device,
                                        weights_only=False)
                if 'loss' in checkpoint:
                    losses.append((path, checkpoint['loss']))
            except Exception as e:
                print(f"Error loading {ckpt}: {e}")
        if losses:
            path = min(losses, key=lambda x: x[1])[0]
        else:
            path = None
    else:
        raise ValueError("Mode must be 'latest' or 'best'")

    # Load and return the selected checkpoint
    if path:
        print(f"Loading checkpoint: {path}")
        # Load the checkpoint and store it in a variable
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Check if checkpoint is a dictionary and print its fields
        if isinstance(checkpoint, dict):
            print("Fields in the checkpoint:", list(checkpoint.keys()))
        else:
            print("Checkpoint is not a dictionary.")
        return checkpoint
    else:
        print("No valid checkpoint found for the specified mode.")
        return None