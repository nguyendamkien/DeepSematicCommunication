#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File: huffman_turbo_test.py
@Description: Tests Huffman + Turbo coding with 64-QAM, with corrected LLRs and stabilized decoding.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import numpy as np
from dataset import EurDataset, collate_pair_data
import huffman
from utils import BleuScore, SNR_to_noise, SeqtoText, Channels, \
    save_evaluation_scores
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from tqdm import tqdm
import argparse
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import normalize as sk_normalize

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Vocabulary loading
VOCAB_FILE = './data/vocab_with_error.json'
vocab = json.load(open(VOCAB_FILE, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
StoT = SeqtoText(token_to_idx, token_to_idx["<END>"])

def strip_special_tokens(text):
    special_tokens = ["<START>", "<END>", "<PAD>", "<UNK>"]
    for token in special_tokens:
        text = text.replace(token, "")
    return ' '.join(text.split())

class HuffmanCoding:
    def __init__(self):
        self.codebook = None
        self.reverse_codebook = None
        self.original_length = 0

    #nén danh sách câu thành chuỗi bit, lưu codebook
    def compress(self, sentences):
        text = ' '.join(sentences)
        freq = {char: text.count(char) for char in set(text)}
        self.codebook = huffman.codebook(freq.items())
        self.reverse_codebook = {v: k for k, v in self.codebook.items()}
        encoded = ''.join(self.codebook[char] for char in text)
        self.original_length = len(encoded)
        return encoded
    
    #chia chuỗi bit thành các đoạn có độ dài tối đa max_bits
    def segment_compress(self, sentence, max_bits):
        compressed = self.compress([sentence])
        segments = [compressed[i:i + max_bits] for i in
                    range(0, len(compressed), max_bits)]
        if len(segments[-1]) < max_bits:
            segments[-1] = segments[-1].ljust(max_bits, '0')
        return segments

    #giải mã chuỗi bit thành text
    def decompress(self, encoded):
        if not self.codebook:
            raise ValueError("Codebook not initialized.")
        decoded, buffer = '', ''
        for bit in encoded[:self.original_length]:
            buffer += bit
            if buffer in self.reverse_codebook:
                decoded += self.reverse_codebook[buffer]
                buffer = ''
        return decoded

    #giải mã các đoạn và ghép lại thành text
    def decompress_segments(self, segments):
        full_encoded = ''.join(segments)
        return self.decompress(full_encoded)

class Modulator:
    def __init__(self, modulation='64qam'):
        self.modulation = modulation
        self.bits_per_symbol = {'64qam': 6}[modulation]
        self.constellation = self._create_constellation()
        self.constellation_points = torch.tensor(
            list(self.constellation.values()), dtype=torch.complex64).to(device)
        self.constellation_keys = list(self.constellation.keys())

    def _create_constellation(self):
        constellation = {f'{i:06b}': (x + y * 1j) for i, (x, y) in
                         enumerate([(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7]
                                    for y in [-7, -5, -3, -1, 1, 3, 5, 7]])}
        points = np.array(list(constellation.values()))
        avg_power = np.mean(np.abs(points) ** 2)
        points = points / np.sqrt(avg_power)
        return {k: points[i] for i, k in enumerate(constellation.keys())}

    def modulate(self, bits):
        if len(bits) % self.bits_per_symbol:
            bits += '0' * (
                    self.bits_per_symbol - len(bits) % self.bits_per_symbol)
        symbols = [self.constellation[bits[i:i + self.bits_per_symbol]] for i in
                   range(0, len(bits), self.bits_per_symbol)]
        return torch.tensor(symbols, dtype=torch.complex64).to(device)

    def demodulate_hard(self, received):
        distances = torch.abs(
            self.constellation_points[None, :] - received[:, None])
        nearest_idx = torch.argmin(distances, dim=1)
        bits = ''.join(
            self.constellation_keys[idx.item()] for idx in nearest_idx)
        return bits

    def demodulate(self, received, noise_std):
        sigma2 = noise_std ** 2
        received = received.to(device)
        distances = torch.abs(
            self.constellation_points[None, :] - received[:, None]) ** 2
        llrs = torch.zeros(len(received) * self.bits_per_symbol, device=device)

        for i in range(self.bits_per_symbol):
            idx_0 = torch.tensor(
                [j for j, k in enumerate(self.constellation_keys) if
                 k[i] == '0'], device=device)
            idx_1 = torch.tensor(
                [j for j, k in enumerate(self.constellation_keys) if
                 k[i] == '1'], device=device)
            min_d0 = torch.min(distances[:, idx_0], dim=1).values
            min_d1 = torch.min(distances[:, idx_1], dim=1).values
            llr = (min_d1 - min_d0) / (2 * sigma2)
            llrs[i::self.bits_per_symbol] = llr
        return llrs

class TurboCoding:
    def __init__(self, block_size=664):
        self.block_size = block_size
        np.random.seed(42)
        self.interleaver = torch.tensor(np.random.permutation(block_size),
                                        dtype=torch.long, device=device)
        self.inv_interleaver = torch.argsort(self.interleaver)
        np.random.seed(None)
        self.memory = 2

    def _convolutional_encode(self, bits):
        shift_register = [0] * self.memory
        systematic = bits
        parity = ''
        for bit in bits:
            bit_int = int(bit)
            parity_bit = (bit_int ^ shift_register[0] ^ shift_register[1]) % 2
            parity += str(parity_bit)
            shift_register = [bit_int] + shift_register[:-1]
        return systematic + parity

    def encode(self, bits):
        self.original_bits = bits
        self.original_len = len(bits)
        if len(bits) > self.block_size:
            bits = bits[:self.block_size]
            print(f"Warning: Truncated input to {self.block_size} bits")
        else:
            bits = bits.ljust(self.block_size, '0')
        enc1 = self._convolutional_encode(bits)
        interleaved_bits = ''.join(
            bits[i] for i in self.interleaver.cpu().numpy())
        enc2 = self._convolutional_encode(interleaved_bits)
        enc2_parity = enc2[self.block_size:]
        return enc1 + enc2_parity

    def _bcjr_decode(self, sys_llrs, parity_llrs, a_priori_llrs=None,
                     use_gpu=True):
        if use_gpu:
            batch_size = sys_llrs.shape[0] if sys_llrs.ndim > 1 else 1
            if batch_size == 1 and sys_llrs.ndim == 1:
                sys_llrs = sys_llrs.unsqueeze(0)
                parity_llrs = parity_llrs.unsqueeze(0)
                a_priori_llrs = a_priori_llrs.unsqueeze(
                    0) if a_priori_llrs is not None else None

            if a_priori_llrs is None:
                a_priori_llrs = torch.zeros((batch_size, self.block_size),
                                            dtype=torch.float32, device=device)
            else:
                a_priori_llrs = torch.as_tensor(a_priori_llrs,
                                                dtype=torch.float32,
                                                device=device)

            sys_llrs = torch.as_tensor(sys_llrs, dtype=torch.float32,
                                       device=device)
            parity_llrs = torch.as_tensor(parity_llrs, dtype=torch.float32,
                                          device=device)

            trellis = [
                [(0, 0, 0), (1, 2, 1)], [(0, 0, 1), (1, 2, 0)],
                [(0, 1, 1), (1, 3, 0)], [(0, 1, 0), (1, 3, 1)]
            ]
            num_states = 4

            alpha = torch.full((batch_size, self.block_size + 1, num_states),
                               float('-inf'), device=device)
            beta = torch.full((batch_size, self.block_size + 1, num_states),
                              float('-inf'), device=device)
            alpha[:, 0, 0] = 0  # Initial state

            branch_metrics = torch.zeros(
                (batch_size, self.block_size, num_states, 2), device=device)
            next_states = torch.tensor([[t[0][1], t[1][1]] for t in trellis],
                                       device=device)  # [4, 2]
            inputs = torch.tensor([[t[0][0], t[1][0]] for t in trellis],
                                  device=device)  # [4, 2]
            parity_out = torch.tensor([[t[0][2], t[1][2]] for t in trellis],
                                      device=device)  # [4, 2]

            # Precompute branch metrics
            for t in range(self.block_size):
                for s in range(num_states):
                    branch_metrics[:, t, s] = (
                            0.5 * a_priori_llrs[:, t].unsqueeze(1) * (
                            1 - 2 * inputs[s]) +
                            0.5 * sys_llrs[:, t].unsqueeze(1) * (
                                    1 - 2 * inputs[s]) +
                            0.5 * parity_llrs[:, t].unsqueeze(1) * (
                                    1 - 2 * parity_out[s])
                    )

            # Forward recursion
            for t in range(self.block_size):
                for s in range(num_states):
                    for inp in range(2):  # 0 or 1
                        next_s = next_states[s, inp]
                        metric = alpha[:, t, s] + branch_metrics[:, t, s, inp]
                        alpha[:, t + 1, next_s] = torch.logsumexp(
                            torch.stack([alpha[:, t + 1, next_s], metric],
                                        dim=1), dim=1
                        )

            # Backward recursion
            beta[:, self.block_size, 0] = 0
            for t in range(self.block_size - 1, -1, -1):
                for s in range(num_states):
                    for inp in range(2):
                        next_s = next_states[s, inp]
                        metric = beta[:, t + 1, next_s] + branch_metrics[:, t,
                                                          s, inp]
                        beta[:, t, s] = torch.logsumexp(
                            torch.stack([beta[:, t, s], metric], dim=1), dim=1
                        )

            # Compute LLRs
            llrs = torch.zeros((batch_size, self.block_size), device=device)
            for t in range(self.block_size):
                p0 = torch.full((batch_size,), float('-inf'), device=device)
                p1 = torch.full((batch_size,), float('-inf'), device=device)
                for s in range(num_states):
                    for inp in range(2):
                        next_s = next_states[s, inp]
                        path_prob = alpha[:, t, s] + branch_metrics[:, t, s,
                                                     inp] + beta[:, t + 1,
                                                            next_s]
                        if inp == 0:
                            p0 = torch.logsumexp(
                                torch.stack([p0, path_prob], dim=1), dim=1)
                        else:
                            p1 = torch.logsumexp(
                                torch.stack([p1, path_prob], dim=1), dim=1)
                llrs[:, t] = p0 - p1

            extrinsic_llrs = llrs - a_priori_llrs - sys_llrs
            return extrinsic_llrs
        else:
            # Ensure inputs are NumPy arrays
            sys_llrs = np.asarray(sys_llrs) if isinstance(sys_llrs,
                                                          torch.Tensor) else sys_llrs
            parity_llrs = np.asarray(parity_llrs) if isinstance(parity_llrs,
                                                                torch.Tensor) else parity_llrs
            a_priori_llrs = np.asarray(a_priori_llrs) if isinstance(
                a_priori_llrs, torch.Tensor) else a_priori_llrs

            if a_priori_llrs is None:
                a_priori_llrs = np.zeros(self.block_size)
            trellis = [
                [(0, 0, 0), (1, 2, 1)], [(0, 0, 1), (1, 2, 0)],
                [(0, 1, 1), (1, 3, 0)], [(0, 1, 0), (1, 3, 1)]
            ]
            num_states = 4

            alpha = np.full((self.block_size + 1, num_states), -np.inf)
            beta = np.full((self.block_size + 1, num_states), -np.inf)
            alpha[0, 0] = 0

            for t in range(self.block_size):
                for s in range(num_states):
                    for inp, next_s, out_p in trellis[s]:
                        branch_metric = (
                                0.5 * a_priori_llrs[t] * (1 - 2 * inp) +
                                0.5 * sys_llrs[t] * (1 - 2 * inp) +
                                0.5 * parity_llrs[t] * (1 - 2 * out_p)
                        )
                        new_alpha = alpha[t, s] + branch_metric
                        alpha[t + 1, next_s] = np.logaddexp(
                            alpha[t + 1, next_s], new_alpha)

            beta[self.block_size, 0] = 0
            for t in range(self.block_size - 1, -1, -1):
                for s in range(num_states):
                    for inp, next_s, out_p in trellis[s]:
                        branch_metric = (
                                0.5 * a_priori_llrs[t] * (1 - 2 * inp) +
                                0.5 * sys_llrs[t] * (1 - 2 * inp) +
                                0.5 * parity_llrs[t] * (1 - 2 * out_p)
                        )
                        new_beta = beta[t + 1, next_s] + branch_metric
                        beta[t, s] = np.logaddexp(beta[t, s], new_beta)

            llrs = np.zeros(self.block_size)
            for t in range(self.block_size):
                p0 = -np.inf
                p1 = -np.inf
                for s in range(num_states):
                    for inp, next_s, out_p in trellis[s]:
                        branch_metric = (
                                0.5 * a_priori_llrs[t] * (1 - 2 * inp) +
                                0.5 * sys_llrs[t] * (1 - 2 * inp) +
                                0.5 * parity_llrs[t] * (1 - 2 * out_p)
                        )
                        path_prob = alpha[t, s] + branch_metric + beta[
                            t + 1, next_s]
                        if inp == 0:
                            p0 = np.logaddexp(p0, path_prob)
                        else:
                            p1 = np.logaddexp(p1, path_prob)
                llrs[t] = p0 - p1

            extrinsic_llrs = llrs - a_priori_llrs - sys_llrs
            return extrinsic_llrs

    def decode(self, llr_batch, iterations=3, use_gpu=True):
        if use_gpu:
            batch_size = llr_batch.shape[0] if llr_batch.ndim > 1 else 1
            if batch_size == 1 and llr_batch.ndim == 1:
                llr_batch = llr_batch.unsqueeze(0)

            sys_llr_batch = llr_batch[:, :self.block_size]
            par1_llr_batch = llr_batch[:, self.block_size:2 * self.block_size]
            par2_llr_batch = llr_batch[:,
                             2 * self.block_size:3 * self.block_size]

            if self.original_len < self.block_size:
                sys_llr_batch[:, self.original_len:] = 10.0

            extrinsic_1to2 = torch.zeros((batch_size, self.block_size),
                                         dtype=torch.float32, device=device)
            extrinsic_2to1 = torch.zeros((batch_size, self.block_size),
                                         dtype=torch.float32, device=device)

            for _ in range(iterations):
                extrinsic_1to2 = self._bcjr_decode(sys_llr_batch,
                                                   par1_llr_batch,
                                                   extrinsic_2to1, use_gpu=True)
                extrinsic_1to2 = torch.clamp(extrinsic_1to2, -20, 20)
                interleaved_sys_llr = sys_llr_batch[:, self.interleaver]
                interleaved_extrinsic = extrinsic_1to2[:, self.interleaver]
                interleaved_extrinsic_2to1 = self._bcjr_decode(
                    interleaved_sys_llr, par2_llr_batch, interleaved_extrinsic,
                    use_gpu=True
                )
                interleaved_extrinsic_2to1 = torch.clamp(
                    interleaved_extrinsic_2to1, -20, 20)
                extrinsic_2to1 = interleaved_extrinsic_2to1[:,
                                 self.inv_interleaver]

            final_llr = sys_llr_batch + extrinsic_2to1
            decoded_batch = torch.where(final_llr > 0, 0, 1)[:,
                            :self.original_len]
            decoded_strings = [''.join(map(str, row.tolist())) for row in
                               decoded_batch]
            return decoded_strings if batch_size > 1 else decoded_strings[0]
        else:
            # Convert GPU tensors to CPU NumPy arrays
            llr_batch = llr_batch.cpu().numpy() if isinstance(llr_batch,
                                                              torch.Tensor) else llr_batch
            sys_llr = llr_batch[:self.block_size]
            par1_llr = llr_batch[self.block_size:2 * self.block_size]
            par2_llr = llr_batch[2 * self.block_size:3 * self.block_size]

            if self.original_len < self.block_size:
                sys_llr[self.original_len:] = 10.0

            extrinsic_1to2 = np.zeros(self.block_size)
            extrinsic_2to1 = np.zeros(self.block_size)

            for _ in range(iterations):
                extrinsic_1to2 = self._bcjr_decode(sys_llr, par1_llr,
                                                   extrinsic_2to1,
                                                   use_gpu=False)
                extrinsic_1to2 = np.clip(extrinsic_1to2, -20, 20)
                interleaved_sys_llr = sys_llr[self.interleaver.cpu().numpy()]
                interleaved_extrinsic = extrinsic_1to2[
                    self.interleaver.cpu().numpy()]
                interleaved_extrinsic_2to1 = self._bcjr_decode(
                    interleaved_sys_llr, par2_llr, interleaved_extrinsic,
                    use_gpu=False
                )
                interleaved_extrinsic_2to1 = np.clip(interleaved_extrinsic_2to1,
                                                     -20, 20)
                extrinsic_2to1 = interleaved_extrinsic_2to1[
                    self.inv_interleaver.cpu().numpy()]

            final_llr = sys_llr + extrinsic_2to1
            decoded = ''.join(
                '0' if l > 0 else '1' for l in final_llr[:self.original_len])
            return decoded

# ── BERT Similarity ───────────────────────────────────────────────────────────
class Similarity:
    def __init__(self, batch_size=4):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def prepare_sentences(self, sentences, max_length=32):
        encoded = self.tokenizer(sentences, padding=True, truncation=True,
                                 max_length=max_length, return_tensors='pt')
        return encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device)

    def get_sentence_embeddings(self, sentences):
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            input_ids, attention_mask = self.prepare_sentences(batch)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True)
                hidden = outputs.hidden_states[-2]
                all_embeddings.append(torch.sum(hidden, dim=1).cpu())
                del outputs, hidden
                torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def compute_similarity(self, real, predicted):
        if not real or not predicted:
            return []
        chunk_size = 100
        all_similarities = []
        total_chunks = (len(real) + chunk_size - 1) // chunk_size
        with tqdm(total=total_chunks, desc="Computing Similarity") as pbar:
            for i in range(0, len(real), chunk_size):
                torch.cuda.empty_cache()
                chunk_real = real[i:i + chunk_size]
                chunk_pred = predicted[i:i + chunk_size]
                real_emb = self.get_sentence_embeddings(chunk_real).cpu().numpy()
                pred_emb = self.get_sentence_embeddings(chunk_pred).cpu().numpy()
                real_norm = sk_normalize(real_emb, norm='max', axis=0)
                pred_norm = sk_normalize(pred_emb, norm='max', axis=0)
                dot = np.sum(real_norm * pred_norm, axis=1)
                a = np.sqrt(np.sum(real_norm ** 2, axis=1))
                b = np.sqrt(np.sum(pred_norm ** 2, axis=1))
                all_similarities.extend((dot / (a * b)).tolist())
                del real_emb, pred_emb, real_norm, pred_norm
                torch.cuda.empty_cache()
                pbar.update(1)
        return all_similarities

def inspect_dataset(dataset, vocab_file='./data/vocab_with_error.json', sample_size=None):
    with open(vocab_file, 'rb') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, end_idx)

    huffman_coder = HuffmanCoding()
    num_samples = sample_size if sample_size else len(dataset)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0,
                            collate_fn=collate_pair_data)
    print(
        f"Inspecting dataset with {len(dataset)} total samples, analyzing {num_samples}...")

    sentence_lengths = []
    compressed_lengths = []

    for i, batch in enumerate(
            tqdm(dataloader, desc="Processing batches", unit="batch")):
        if i * 32 >= num_samples:
            break
        noise_batch, trg_batch, label_batch = batch
        input_sentences = [StoT.sequence_to_text(sent.tolist()) for sent in
                           noise_batch.cpu()]
        for sent in input_sentences:
            cleaned_sent = strip_special_tokens(sent)
            tokens = cleaned_sent.split()
            sentence_lengths.append(len(tokens))
            if tokens:
                compressed_bits = huffman_coder.compress([cleaned_sent])
                compressed_lengths.append(len(compressed_bits))

    total_sentences = len(sentence_lengths)
    if total_sentences == 0:
        print("No valid sentences found in dataset.")
        return None

    total_tokens = sum(sentence_lengths)
    avg_sentence_length = np.mean(sentence_lengths)
    sentences_in_range = sum(
        1 for length in sentence_lengths if 4 <= length <= 30)
    avg_compressed_length = np.mean(compressed_lengths)
    max_compressed_length = max(compressed_lengths)
    min_compressed_length = min(compressed_lengths)
    percentile_95_compressed = np.percentile(compressed_lengths, 95)

    print(
        f"\nDataset Inspection Results (based on {total_sentences} sentences):")
    print(f"Total tokens: {total_tokens}")
    print(f"Average sentence length (tokens): {avg_sentence_length:.1f}")
    print(
        f"Sentences with 4-30 tokens: {sentences_in_range} ({sentences_in_range / total_sentences * 100:.1f}%)")
    print(
        f"Average Huffman-compressed length (bits): {avg_compressed_length:.1f}")
    print(f"Max Huffman-compressed length (bits): {max_compressed_length}")
    print(f"Min Huffman-compressed length (bits): {min_compressed_length}")
    print(
        f"95th percentile compressed length (bits): {percentile_95_compressed:.1f}")

    suggested_k = int(np.ceil(percentile_95_compressed / 8))
    suggested_n = min(255, suggested_k + 60)
    if suggested_n <= suggested_k:
        suggested_n = min(255, suggested_k + 30)
    print(f"\nSuggested RS parameters:")
    print(
        f" - k = {suggested_k} (covers 95% of compressed lengths, {suggested_k * 8} bits)")
    print(
        f" - n = {suggested_n} (error correction up to {(suggested_n - suggested_k) // 2} symbols)")

    return {
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "avg_sentence_length": avg_sentence_length,
        "sentences_in_range": sentences_in_range,
        "avg_compressed_length": avg_compressed_length,
        "max_compressed_length": max_compressed_length,
        "min_compressed_length": min_compressed_length,
        "percentile_95_compressed": percentile_95_compressed,
        "suggested_n": suggested_n,
        "suggested_k": suggested_k
    }


def debug_huffman_turbo(dataloader: DataLoader, snr: float, modulation: str,
                        bleu_ngrams: list, block_size=664, num_samples=10):
    huffman_coder = HuffmanCoding()
    turbo_coder = TurboCoding(block_size=block_size)
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    similarity = Similarity()
    print(
        f"\nDebugging {len(dataloader.dataset)} sentence(s) with Huffman + Turbo "
        f"(block_size={block_size}) ({modulation}) at SNR {snr} dB")

    bleu_calcs = {n: BleuScore(*[1 / n if i < n else 0 for i in range(4)]) for n
                  in bleu_ngrams}
    total_sentences = 0
    configs = [
        {"iterations": 1, "use_gpu": True, "desc": "GPU Optimized (1 iter)"},
        {"iterations": 1, "use_gpu": False, "desc": "CPU (1 iter)"}
    ]
    results = {config["desc"]: {"time": 0, "bleu_scores": []} for config in
               configs}

    for noise_batch, clean_batch, label_batch in dataloader:
        if isinstance(noise_batch, torch.Tensor) and noise_batch.ndim > 1:
            noise_sent = noise_batch[0]
        else:
            noise_sent = noise_batch

        if isinstance(clean_batch, torch.Tensor) and clean_batch.ndim > 1:
            target_sent = clean_batch[0]
        else:
            target_sent = clean_batch

        noise_text = StoT.sequence_to_text(noise_sent.tolist())
        target_text = StoT.sequence_to_text(target_sent.tolist())

        cleaned_noise = strip_special_tokens(noise_text)
        cleaned_target = strip_special_tokens(target_text)

        if total_sentences >= num_samples:
            break

        compressed = huffman_coder.compress([cleaned_noise])
        print(f"\nSample {total_sentences + 1}:")
        print(f"Noise input: {cleaned_noise}")
        print(f"Target ref: {cleaned_target}")
        print(
            f"Compressed bits: {compressed[:50]}... (len={len(compressed)})")

        encoded_bits = turbo_coder.encode(compressed)
        print(
            f"Encoded bits: {encoded_bits[:50]}... (len={len(encoded_bits)})")

        symbols = modulator.modulate(encoded_bits)
        symbols_ri = torch.stack((symbols.real, symbols.imag), dim=-1)
        noise_std = SNR_to_noise(snr)
        rx_sig_tensor, _ = channel.Rician(symbols_ri, noise_std)
        received = rx_sig_tensor[:, 0] + 1j * rx_sig_tensor[:, 1]

        hard_bits = modulator.demodulate_hard(received)
        hard_errors = sum(a != b for a, b in
                          zip(encoded_bits, hard_bits[:len(encoded_bits)]))
        print(f"Hard decision errors: {hard_errors}/{len(encoded_bits)}")

        llr = modulator.demodulate(received, noise_std)

        for config in configs:
            start_time = time.time()
            decoded_bits = turbo_coder.decode(llr, iterations=config[
                    "iterations"], use_gpu=config["use_gpu"])
            end_time = time.time()
            elapsed = end_time - start_time

            output_text = huffman_coder.decompress(decoded_bits) or ""
            bleu_score = bleu_calcs[1].compute_blue_score([cleaned_target],
                                                              [output_text])[0]
            sim = similarity.compute_similarity([cleaned_target], [output_text])[0]

            results[config["desc"]]["time"] += elapsed
            results[config["desc"]]["bleu_scores"].append(bleu_score)

            print(f"\n{config['desc']}:" )
            print(
                f"Decoded bits: {decoded_bits[:50]}... (len={len(decoded_bits)})")
            print(f"Output: {output_text}")
            print(f"Time: {elapsed:.3f}s, BLEU-1: {bleu_score:.4f}, Sim: {sim:.4f}")

        total_sentences += 1
        if total_sentences >= num_samples:
            break
        torch.cuda.empty_cache()
        if total_sentences >= num_samples:
            break

    print(f"\nProcessed {total_sentences} sentence(s)")
    for desc, res in results.items():
        avg_bleu = np.mean(res["bleu_scores"])
        total_time = res["time"]
        print(
            f"{desc}: Avg BLEU-1 = {avg_bleu:.4f}, Total Time = {total_time:.3f}s")

    return results


def batch_test_huffman_turbo(dataloader: DataLoader, snrs: list,
                             modulation: str, args,
                             block_size=688, use_gpu=True) -> None:
    huffman_coder = HuffmanCoding()
    turbo_coder = TurboCoding(block_size=block_size)
    modulator = Modulator(modulation=modulation)
    channel = Channels()
    similarity = Similarity()

    bleu_calc = BleuScore(1, 0, 0, 0)   # BLEU-1

    final_bleu = []
    final_sim = []

    for snr in snrs:
        print(f"\nProcessing SNR: {snr} dB")
        noise_std = SNR_to_noise(snr)
        total_bleu = 0.0
        total_sentences = 0
        total_similarity = 0.0
        total_items = len(dataloader.dataset)

        truncated_sentences = 0
        total_ber_pre_decode = 0.0
        total_items = len(dataloader.dataset)

        start_time = time.time()

        with tqdm(total=total_items,
                desc=f"Batch Test SNR {snr} dB (GPU: {use_gpu})",
                leave=True) as pbar:
            for  noise_batch, clean_batch, label_batch  in dataloader:

                batch_size = noise_batch.size(0)

                for i in range(batch_size):

                    noise_sent = noise_batch[i]
                    target_sent = clean_batch[i]
    
                    cleaned_noise  = strip_special_tokens(StoT.sequence_to_text(noise_sent.tolist()))
                    cleaned_target = strip_special_tokens(StoT.sequence_to_text(target_sent.tolist()))
        

                    # Huffman encoding
                    compressed = huffman_coder.compress([cleaned_noise])
                    original_length = min(len(compressed), block_size)
                    truncated = len(compressed) > block_size
                    truncated_sentences += truncated
                    compressed = compressed[
                                :block_size] if truncated else compressed.ljust(
                        block_size, '0')

                    # Turbo encoding
                    turbo_coder.original_len = original_length
                    encoded_bits = turbo_coder.encode(compressed)

                    # Modulation and channel
                    symbols = modulator.modulate(encoded_bits)
                    symbols_ri = torch.stack((symbols.real, symbols.imag), dim=-1)
                    noise_std = SNR_to_noise(snr)
                    rx_sig_tensor, _ = channel.Rician(symbols_ri, noise_std)
                    received = rx_sig_tensor[:, 0] + 1j * rx_sig_tensor[:, 1]

                    # Hard decision for pre-decode BER
                    hard_bits = modulator.demodulate_hard(received)
                    pre_decode_errors = sum(a != b for a, b in zip(encoded_bits,
                                                                hard_bits[
                                                                :len(encoded_bits)]))
                    total_ber_pre_decode += pre_decode_errors / len(encoded_bits)

                    # Soft demodulation and turbo decoding
                    llr = modulator.demodulate(received, noise_std)
                    decoded_bits = turbo_coder.decode(llr, iterations=1,
                                                    use_gpu=use_gpu)
                    output_text = huffman_coder.decompress(decoded_bits) or ""

                    similarity_score = similarity.compute_similarity(
                        [cleaned_target],
                        [output_text]
                    )[0] if output_text else 0.0

                    total_similarity += similarity_score

                    total_bleu += bleu_calc.compute_blue_score(
                                                [cleaned_target],
                                                [output_text]
                                            )[0]

                    # Debug output for first 10 samples
                    if total_sentences < 10:
                        print(f"\nSample {total_sentences + 1}:")
                        print(f"Input: {cleaned_noise[:50]}...")
                        print(
                            f"Compressed: {compressed[:50]}... (len={len(compressed)})")
                        print(
                            f"Encoded: {encoded_bits[:50]}... (len={len(encoded_bits)})")
                        print(
                            f"Hard decision errors: {pre_decode_errors}/{len(encoded_bits)}")
                        print(
                            f"Decoded: {decoded_bits[:50]}... (len={len(decoded_bits)})")
                        print(f"Output: {output_text[:50]}...")
                        print(f"clean target: {cleaned_target[:50]}...")
                        print(f"Similarity: {similarity_score:.4f}")

                    total_sentences += 1
                    pbar.update(1)
                if use_gpu:
                    torch.cuda.empty_cache()

        end_time = time.time()
        runtime = end_time - start_time

        avg_bleu = total_bleu / total_sentences
        avg_similarity = total_similarity / total_sentences

        final_bleu.append(avg_bleu)
        final_sim.append(avg_similarity)

        print(f"\nProcessed {total_sentences} sentences in {runtime:.2f} seconds")
        print(f"\n[SNR {snr:+.1f} dB] {total_sentences} câu | {runtime:.1f}s")
        print(f"  BLEU={avg_bleu:.4f} | Similarity={avg_similarity:.4f}")

        save_evaluation_scores(args, [snr], [avg_bleu], [avg_similarity],
                                method="Huffman-Turbo", bleu_ngram=1)
            
    save_evaluation_scores(args, snrs, final_bleu, final_sim, 'Huffman-Turbo', 1)
    return final_bleu, final_sim
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huffman + Turbo Test Script")
    parser.add_argument('--channel', default='Rician', type=str,
                        help='Channel type')
    parser.add_argument('--MAX-LENGTH', default=30, type=int,
                        help='Maximum sentence length')
    parser.add_argument('--MIN-LENGTH', default=4, type=int,
                        help='Minimum sentence length')
    parser.add_argument('--d-model', default=128, type=int,
                        help='Model dimension (placeholder)')
    parser.add_argument('--num-layers', default=4, type=int,
                        help='Number of layers (placeholder)')
    parser.add_argument('--num-heads', default=8, type=int,
                        help='Number of heads (placeholder)')
    parser.add_argument('--dff', default=512, type=int,
                        help='Feed-forward dimension (placeholder)')
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU for decoding (default: True)')
    args = parser.parse_args()

    test_eur = EurDataset('test')
    snrs = [0, 3, 6, 9, 12, 15, 18]
    modulation = '64qam'
    bleu_ngrams = [1]
    use_gpu = True  # Set to False to use CPU

    print("Running dataset inspection...")
    stats = inspect_dataset(test_eur, sample_size=len(test_eur))
    block_size = 688
    if stats:
        block_size = int(
            np.ceil(stats.get('percentile_95_compressed', 680.8) / 8) * 8)
        print(f"Using block size: {block_size} bits")

    # Create a subset of ~1000 samples by taking every 140th sample
    total_samples = len(test_eur)  # 140,000
    subset_size = 100
    step = total_samples // subset_size  # ~140
    subset_indices = list(range(0, total_samples, step))[
                     :subset_size]  # Take first 1000 steps
    print(
        f"\nTesting on a subset of {len(subset_indices)} samples (every {step}th sample from {total_samples})")

    from torch.utils.data import Subset

    # subset = Subset(test_eur, subset_indices)
    # batch_loader = DataLoader(subset, batch_size=128, num_workers=4,
    #                           collate_fn=collate_pair_data)

    # batch_test_huffman_turbo(batch_loader, snrs, modulation,
    #                              args, block_size, use_gpu=use_gpu)

    "Test sample"

    from torch.utils.data import Subset
    subset = Subset(test_eur, [0, 1, 2, 3, 86]) # lấy index 86
    debug_loader = DataLoader(subset, batch_size=1, collate_fn=collate_pair_data)

    debug_huffman_turbo(debug_loader, snr=18, modulation=modulation,
                        bleu_ngrams=bleu_ngrams, block_size=block_size,
                        num_samples=5)

