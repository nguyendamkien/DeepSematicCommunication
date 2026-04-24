import argparse
import json
import os
import random
import signal
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_pair_data
# from models.mutual_info import Mine
from models.transceiver_calibration import CA_DeepSC
from utils import SNR_to_noise, train_step_calibration, val_step_calibration, initNetParams, \
    SeqtoText, list_checkpoints, load_checkpoint, create_masks

plt.ion() # Turn on interactive mode

# Argument parser for configuring hyperparameters and paths
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='vocab_with_error.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='/kaggle/working/checkpoints/ca-deepsc-AWGN',
                    type=str)
parser.add_argument('--channel', default='AWGN', type=str,
                    help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=30, type=int)

# thêm argument action
parser.add_argument(
    "--action",
    choices=["start", "resume"],
    default="start",
    help="Choose 'start' to train from scratch or 'resume' to continue from checkpoint"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stop_training = False  # Global variable to control training interruption

# Signal handler for stopping training gracefully
def signal_handler(sig, frame):
    global stop_training
    print("\nTraining interruption signal received. Saving progress...")
    stop_training = True

# Function to set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Training funtion
# def train(epoch, args, net, mi_net=None):
#     global stop_training
#     train_eur = EurDataset('train')
#     train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
#                                 num_workers=0, pin_memory=True,
#                                 collate_fn=collate_pair_data)
#     pbar = tqdm(train_iterator)

#     # For TimeVaryingRician
#     # noise_std_options = np.arange(0.045, 0.316, 0.010)
#     epoch_loss = 0
#     epoch_bce_loss = 0
#     mi_bits_total = 0
#     batch_count = 0
#     snr_values = []

#     #noise_sent la cau goc, clean la target muon nhan
#     for noise_sents, clean_sents, labels in pbar:
#         if stop_training:
#             return True, epoch_loss, epoch_bce_loss, mi_bits_total / batch_count if batch_count > 0 else 0, min(
#                 snr_values) if snr_values else 0, max(
#                 snr_values) if snr_values else 0, sum(snr_values) / len(
#                 snr_values) if snr_values else 0
#         noise_sents = noise_sents.to(device)
#         clean_sents = clean_sents.to(device)
#         labels = labels.to(device)
#         # noise_std = np.random.choice(noise_std_options, size=1).item()  # Scalar
#         # For original Channel
#         noise_std = float(
#             np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0])
#         # if mi_net is not None:
#         #     mi_loss, mi_bits = train_mi(net, mi_net, sents, 0.1, pad_idx,
#         #                                 mi_opt, args.channel)
#         #     loss_total, snr = train_step(net, sents, sents, 0.1, pad_idx,
#         #                                  optimizer, criterion, args.channel,
#         #                                  mi_net)
#         #     epoch_loss += loss_total
#         #     mi_bits_total += mi_bits
#         #     batch_count += 1
#         #     snr_values.append(snr)
#         #     pbar.set_description(
#         #         f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; MI Loss: {mi_loss:.5f}; MI (bits): {mi_bits:.5f}; SNR: {snr:.5f}')
#         # else:
#         loss_total, bce_loss_val, snr = train_step_calibration(net, noise_sents, clean_sents, labels, noise_std, pad_idx,
#                                      optimizer, criterion, args.channel, bce_loss_fn)
#         epoch_loss += loss_total
#         epoch_bce_loss += bce_loss_val
#         snr_values.append(snr)
#         pbar.set_description(
#             f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; BCE: {bce_loss_val:.5f}; SNR: {snr:.5f}; Noise Std: {noise_std:.5f}')

#     snr_min = min(snr_values) if snr_values else 0
#     snr_max = max(snr_values) if snr_values else 0
#     snr_avg = sum(snr_values) / len(snr_values) if snr_values else 0

#     avg_epoch_loss = epoch_loss / len(train_iterator)
#     avg_epoch_bce_loss = epoch_bce_loss / len(train_iterator)
#     avg_mi_bits = mi_bits_total / batch_count if batch_count > 0 else 0
#     return False, avg_epoch_loss, avg_epoch_bce_loss, avg_mi_bits, snr_min, snr_max, snr_avg

def train(epoch, args, net, mi_net=None):
    global stop_training
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                num_workers=0, pin_memory=True,
                                collate_fn=collate_pair_data)
    pbar = tqdm(train_iterator)
    epoch_loss = 0
    epoch_bce_loss = 0
    mi_bits_total = 0
    batch_count = 0
    snr_values = []

    # # ===== DEBUG 3 MẪU =====
    # net.eval()
    # with torch.no_grad():
    #     for noise_sents, clean_sents, labels in train_iterator:
    #         noise_sents = noise_sents.to(device)
    #         labels = labels.to(device)

    #         src_mask, _ = create_masks(noise_sents, noise_sents[:, :-1], pad_idx)
    #         _, pred_error_prob = net.encoder(noise_sents, src_mask)

    #         # Chỉ in 3 mẫu đầu
    #         for i in range(min(3, noise_sents.size(0))):
    #             tokens = noise_sents[i].cpu().numpy()
    #             probs  = pred_error_prob[i].cpu().numpy()
    #             labs   = labels[i].cpu().numpy()

    #             # Convert ids về words
    #             words = [seq_to_text.reverse_word_map.get(t, '<UNK>') 
    #                      for t in tokens if t != pad_idx]
    #             probs = probs[:len(words)]
    #             labs  = labs[:len(words)]

    #             print(f"\n===== Mẫu {i+1} =====")
    #             print(f"{'Token':<15} {'True':>6} {'Pred':>8} {'OK?':>5}")
    #             print("-" * 40)
    #             for w, l, p in zip(words, labs, probs):
    #                 ok = "✅" if (p > 0.5) == (l == 1) else "❌"
    #                 print(f"{w:<15} {int(l):>6} {p:>8.4f} {ok:>5}")

    #             # Tóm tắt
    #             pred_binary = (probs > 0.5).astype(float)
    #             correct = (pred_binary == labs).mean()
    #             sep = probs[labs == 1].mean() - probs[labs == 0].mean() \
    #                   if labs.sum() > 0 else float('nan')
    #             print(f"Accuracy: {correct:.3f} | Separation: {sep:.3f}")
    #         break  # Chỉ debug 1 batch
    # # ===== END DEBUG =====

    # ===== DEBUG ATTENTION WEIGHTS =====
    # net.eval()
    # with torch.no_grad():
    #     for noise_sents, clean_sents, labels in train_iterator:
    #         noise_sents = noise_sents.to(device)
    #         labels = labels.to(device)

    #         src_mask, _ = create_masks(noise_sents, noise_sents[:, :-1], pad_idx)
    #         _, pred_error_prob = net.encoder(noise_sents, src_mask)
    #         layer = net.encoder.layers[-1].cattn


    #         for i in range(min(1, noise_sents.size(0))):
    #             tokens = noise_sents[i].cpu().numpy()
    #             probs  = pred_error_prob[i].cpu().numpy()
    #             labs   = labels[i].cpu().numpy()

    #             words = [seq_to_text.reverse_word_map.get(t, '<UNK>') 
    #                      for t in tokens if t != pad_idx]
    #             probs = probs[:len(words)]
    #             labs  = labs[:len(words)]

    #             # ===== Calibration matrix C =====
    #             P = torch.tensor(probs)
    #             P_outer = torch.ger(P, P)       # [seq_len, seq_len]
    #             C = 1 - P_outer                 # [seq_len, seq_len]
                
    #             # C = (1 - P).unsqueeze(1).unsqueeze(2)   # [batch,1,1,seq]
    #             # C = C.expand(-1, 1, P.size(1), -1)   

    #             print(f"\n===== Mẫu {i+1} =====")
    #             print(f"{'Token':<15} {'True':>6} {'Pred':>8} {'Weight(1-P)':>12}")
    #             print("-" * 50)
    #             for w, l, p in zip(words, labs, probs):
    #                 print(f"{w:<15} {int(l):>6} {p:>8.4f} {1-p:>12.4f}")

    #             # C matrix — chỉ in diagonal và một số cặp quan trọng
    #             print(f"\n--- Calibration C (token lỗi vs token đúng) ---")
    #             error_indices = [j for j, l in enumerate(labs) if l == 1]
    #             clean_indices = [j for j, l in enumerate(labs) if l == 0]

    #             if error_indices and clean_indices:
    #                 ei = error_indices[0]   # token lỗi đầu tiên
    #                 ci = clean_indices[0]   # token đúng đầu tiên

    #                 print(f"C[clean→clean] ({words[ci]}→{words[ci]}): {C[ci,ci]:.4f}")
    #                 print(f"C[error→error] ({words[ei]}→{words[ei]}): {C[ei,ei]:.4f}")
    #                 print(f"C[clean→error] ({words[ci]}→{words[ei]}): {C[ci,ei]:.4f}")
    #                 print(f"C[error→clean] ({words[ei]}→{words[ci]}): {C[ei,ci]:.4f}")

    #                 print(f"\n→ Token lỗi '{words[ei]}' attend đến token đúng '{words[ci]}'")
    #                 print(f"  với weight {C[ei,ci]:.4f} (thấp = bị giảm ảnh hưởng)")

    #             # So sánh attention scores trước và sau calibration
    #             print(f"\n--- Ảnh hưởng C lên attention scores ---")
    #             # Giả sử scores uniform trước calibration
    #             seq_len = len(words)
                
    #             attn_before = layer.attn_before[i].mean(0)  # [seq, seq]
    #             attn_after  = layer.attn_after[i].mean(0)

    #             print(f"{'Token':<15} {'Attn trước':>12} {'Attn sau':>12} {'Thay đổi':>10}")
    #             print("-" * 55)
    #             for j, w in enumerate(words):
    #                 before = attn_before[3, j].item()
    #                 after  = attn_after[3, j].item()
    #                 delta  = after - before
    #                 flag   = "↓ lỗi" if labs[j] == 1 else ""
    #                 print(f"{w:<15} {before:>12.4f} {after:>12.4f} {delta:>+10.4f} {flag}")

    #             print(f"\n--- Ma trận attention đầy đủ [query × key] ---")
    #             print(f"{'':>12}", end="")
    #             for w in words:
    #                 print(f"{w[:6]:>8}", end="")
    #             print()
    #             for r, rw in enumerate(words):
    #                 flag = "←err" if labs[r]==1 else ""
    #                 print(f"{rw[:11]:>11}{flag[:4]} │", end="")
    #                 for c in range(len(words)):
    #                     before = attn_before[r,c].item()
    #                     after  = attn_after[r,c].item()
    #                     delta  = after - before
    #                     marker = "↓" if labs[c]==1 and abs(delta)>0.005 else " "
    #                     print(f"{after:>7.4f}{marker}", end="")           
    #         break

    net.eval()
    with torch.no_grad():
        for noise_sents, trg_sents, label_tensors in train_iterator:
            noise_sents = noise_sents.to(device)
            label_tensors = label_tensors.to(device)
            src_mask, _ = create_masks(noise_sents, noise_sents[:, :-1], pad_idx)

            enc_output, P = net.encoder(noise_sents, src_mask)

            # Lấy layer cuối của CA-DeepSC
            last_layer = net.encoder.layers[-1].cattn
            pred_error_prob = P
            # layer = net.encoder.layers[-1].cattn

            for i in range(min(1, noise_sents.size(0))):
                tokens = noise_sents[i].cpu().numpy()
                probs  = pred_error_prob[i].cpu().numpy()
                labs   = label_tensors[i].cpu().numpy()

                words = [seq_to_text.reverse_word_map.get(t, '<UNK>') 
                         for t in tokens if t != pad_idx]
                probs = probs[:len(words)]
                labs  = labs[:len(words)]

                # ===== Calibration matrix C =====
                P_pred = torch.tensor(probs)
                P_outer = torch.ger(P_pred, P_pred)       # [seq_len, seq_len]
                C = 1 - P_outer                 # [seq_len, seq_len]
                
                # C = (1 - P).unsqueeze(1).unsqueeze(2)   # [batch,1,1,seq]
                # C = C.expand(-1, 1, P.size(1), -1)   

                print(f"\n===== Mẫu {i+1} =====")
                print(f"{'Token':<15} {'True':>6} {'Pred':>8} {'Weight(1-P)':>12}")
                print("-" * 50)
                for w, l, p in zip(words, labs, probs):
                    print(f"{w:<15} {int(l):>6} {p:>8.4f} {1-p:>12.4f}")

                # C matrix — chỉ in diagonal và một số cặp quan trọng
                print(f"\n--- Calibration C (token lỗi vs token đúng) ---")
                error_indices = [j for j, l in enumerate(labs) if l == 1]
                clean_indices = [j for j, l in enumerate(labs) if l == 0]

                if error_indices and clean_indices:
                    ei = error_indices[0]   # token lỗi đầu tiên
                    ci = clean_indices[0]   # token đúng đầu tiên

                    print(f"C[clean→clean] ({words[ci]}→{words[ci]}): {C[ci,ci]:.4f}")
                    print(f"C[error→error] ({words[ei]}→{words[ei]}): {C[ei,ei]:.4f}")
                    print(f"C[clean→error] ({words[ci]}→{words[ei]}): {C[ci,ei]:.4f}")
                    print(f"C[error→clean] ({words[ei]}→{words[ci]}): {C[ei,ci]:.4f}")

                    print(f"\n→ Token lỗi '{words[ei]}' attend đến token đúng '{words[ci]}'")
                    print(f"  với weight {C[ei,ci]:.4f} (thấp = bị giảm ảnh hưởng)")

                # So sánh attention scores trước và sau calibration
                print(f"\n--- Ảnh hưởng C lên attention scores ---")
                # Giả sử scores uniform trước calibration
                seq_len = len(words)
                
                attn_before = last_layer.attn_before[i].mean(0)  # [seq, seq]
                attn_after  = last_layer.attn_after[i].mean(0)

                print(f"{'Token':<15} {'Attn trước':>12} {'Attn sau':>12} {'Thay đổi':>10}")
                print("-" * 55)
                for j, w in enumerate(words):
                    before = attn_before[3, j].item()
                    after  = attn_after[3, j].item()
                    delta  = after - before
                    flag   = "↓ lỗi" if labs[j] == 1 else ""
                    print(f"{w:<15} {before:>12.4f} {after:>12.4f} {delta:>+10.4f} {flag}")

                # print(f"\n--- Ma trận attention đầy đủ [query × key] ---")
                # print(f"{'':>12}", end="")
                # for w in words:
                #     print(f"{w[:6]:>8}", end="")
                # print()
                # for r, rw in enumerate(words):
                #     flag = "←err" if labs[r]==1 else ""
                #     print(f"{rw[:11]:>11}{flag[:4]} │", end="")
                #     for c in range(len(words)):
                #         before = attn_before[r,c].item()
                #         after  = attn_after[r,c].item()
                #         delta  = after - before
                #         marker = "↓" if labs[c]==1 and abs(delta)>0.005 else " "
                #         print(f"{after:>7.4f}{marker}", end="")  

            print("\n" + "=" * 65)
            i = 0  # câu đầu tiên
            tokens = noise_sents[i].cpu().numpy()
            words = [seq_to_text.reverse_word_map.get(int(t), '<UNK>')
                    for t in tokens if int(t) != pad_idx]
            seq_len = len(words)

            def print_matrix(mat, row_labels, col_labels, title, fmt=".3f"):
                """In ma trận dạng ASCII với shade ký tự"""
                col_w = 9
                print(f"\n{'='*65}")
                print(f"  {title}")
                print(f"  shape={list(mat.shape)}  "
                      f"std={mat.std().item():.5f}  "
                      f"mean={mat.mean().item():.5f}")
                print(f"{'='*65}")
                print(f"{'':>12}", end="")
                for w in col_labels:
                    print(f"{w[:col_w-1]:>{col_w}}", end="")
                print()
                print(f"{'':>12}" + "─"*(col_w*len(col_labels)))

                mx = mat.abs().max().item() + 1e-9
                shades = ["   ", "░░░", "▒▒▒", "▓▓▓", "███"]

                for r, rw in enumerate(row_labels):
                    print(f"{rw[:11]:>11} │", end="")
                    row_sum = 0
                    for c in range(len(col_labels)):
                        v = mat[r, c].item()
                        row_sum += v
                        lvl = min(4, int(abs(v) / mx * 4.99))
                        cell = f"{shades[lvl]}{v:{fmt}}"
                        print(f" {cell}"[:col_w], end="")
                    print(f"  Σ={row_sum:.3f}")
                print(f"{'':>12}" + "─"*(col_w*len(col_labels)))

            # ── Ma trận 1: p_attn TRƯỚC calibration ─────────────────────
            # attn_before = last_layer.attn_before[i].mean(0)[:seq_len, :seq_len].cpu()
            # print_matrix(attn_before, words, words,
            #             "p_attn BEFORE calibration = softmax(QKᵀ/√dk)")

            # ── Ma trận 2: p_attn SAU calibration ────────────────────────
            # attn_after = last_layer.attn_after[i].mean(0)[:seq_len, :seq_len].cpu()
            # print_matrix(attn_after, words, words,
            #             "p_attn AFTER calibration  = softmax(·) × C  ← sum != 1 nữa")

            # ── Ma trận 3: context = p_attn_after × V ────────────────────
            ctx = last_layer.context[i].cpu()          # [heads, seq, d_k]
            ctx_mean = ctx.mean(0)[:seq_len, :]        # [seq, d_k]
            show_dims = min(ctx_mean.size(1), 16)
            dim_labels = [f"d{j}" for j in range(show_dims)]
            print_matrix(ctx_mean[:, :show_dims], words, dim_labels,
                        f"context = p_attn_after × V  (mean {ctx.size(0)} heads, {show_dims} dims đầu)",
                        fmt=".2f")

            # ── L2 norm của context ───────────────────────────────────────
            ctx_norm = ctx_mean.norm(dim=-1)
            print(f"\n--- ‖context‖ L2 norm ---")
            print(f"{'Token':<15} {'norm':>7}  bar")
            print("─"*52)
            mx_norm = ctx_norm.max().item() + 1e-9
            for j, w in enumerate(words):
                nv = ctx_norm[j].item()
                bar = "█" * int(nv/mx_norm*30) + "░"*(30-int(nv/mx_norm*30))
                print(f"{w:<15} {nv:>7.4f}  {bar}")

            # ── P(error) và C diagonal ────────────────────────────────────
            p_i = P[i, :seq_len].cpu()
            print(f"\n--- P(error) và C[j,j] = 1 - P[j]² (diagonal của C) ---")
            print(f"{'Token':<15} {'P(err)':>8}  {'C[j,j]':>8}  bar P")
            print("─"*55)
            for j, w in enumerate(words):
                pv = p_i[j].item()
                cjj = 1 - pv*pv
                bar = "█"*int(pv*30) + "░"*(30-int(pv*30))
                flag = " ← HIGH" if pv > 0.4 else ""
                print(f"{w:<15} {pv:>8.4f}  {cjj:>8.4f}  {bar}{flag}")

            # ── Std từng head: before vs after ───────────────────────────
            print(f"\n--- Std từng head (before → after calibration) ---")
            for h in range(last_layer.attn_before[i].size(0)):
                sb = last_layer.attn_before[i, h, :seq_len, :seq_len].std().item()
                sa = last_layer.attn_after[i,  h, :seq_len, :seq_len].std().item()
                flag_b = " ← uniform!" if sb < 1e-4 else ""
                print(f"  Head {h}: {sb:.6f} → {sa:.6f}{flag_b}")

            break
    net.train()
    for noise_sents, clean_sents, labels in pbar:
        if stop_training:
            return True, epoch_loss, epoch_bce_loss, \
                   mi_bits_total / batch_count if batch_count > 0 else 0, \
                   min(snr_values) if snr_values else 0, \
                   max(snr_values) if snr_values else 0, \
                   sum(snr_values) / len(snr_values) if snr_values else 0

        noise_sents = noise_sents.to(device)
        clean_sents = clean_sents.to(device)
        labels = labels.to(device)

        noise_std = float(np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0])
        loss_total, bce_loss_val, snr = train_step_calibration(
            net, noise_sents, clean_sents, labels, noise_std, pad_idx,
            optimizer, criterion, args.channel, bce_loss_fn)

        epoch_loss += loss_total
        epoch_bce_loss += bce_loss_val
        snr_values.append(snr)
        pbar.set_description(
            f'Epoch: {epoch+1}; Loss: {loss_total:.5f}; BCE: {bce_loss_val:.5f}; SNR: {snr:.5f}')

    snr_min = min(snr_values) if snr_values else 0
    snr_max = max(snr_values) if snr_values else 0
    snr_avg = sum(snr_values) / len(snr_values) if snr_values else 0

    avg_epoch_loss    = epoch_loss / len(train_iterator)
    avg_epoch_bce_loss = epoch_bce_loss / len(train_iterator)
    avg_mi_bits       = mi_bits_total / batch_count if batch_count > 0 else 0

    return False, avg_epoch_loss, avg_epoch_bce_loss, avg_mi_bits, snr_min, snr_max, snr_avg

# Validation function
def validate(epoch, args, net, seq_to_text):
    val_eur = EurDataset('val')  # Load val dataset
    val_iterator = DataLoader(val_eur, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_pair_data)

    # # Print a sample batch from test_iterator for debugging
    # sample_batch = next(iter(test_iterator))  # Get first batch
    # print("Sample batch from test_iterator (Tensor format):")
    # print(sample_batch)  # Print the raw tensor

    # # Convert token IDs to text using sequence_to_text method
    # decoded_sentences = [
    #     seq_to_text.sequence_to_text(sent.cpu().numpy().tolist()) for sent in
    #     sample_batch]
    # print("\nSample batch (Decoded sentences):")
    # for i, sent in enumerate(decoded_sentences):
    #     print(f"Sentence {i + 1}: {sent}")

    # print_padded_sentences(test_iterator, seq_to_text, pad_idx)

    net.eval()
    pbar = tqdm(val_iterator)
    total = 0
    total_bce = 0
    # Noise_std for TimeVaryingRician
    # noise_std_options = np.arange(0.045, 0.316, 0.010)
    # noise_std = np.random.choice(noise_std_options, size=1)
    with torch.no_grad():
        for noise_sents, clean_sents, labels in pbar:
            # print(f"Batch contains {sents.shape[0]} sentences")
            noise_sents = noise_sents.to(device)
            clean_sents = clean_sents.to(device)
            labels = labels.to(device)
            loss, bce_loss_val, snr = val_step_calibration(net, noise_sents, clean_sents, labels, 0.1, pad_idx, criterion,
                                 args.channel, bce_loss_fn)
            # TimeVaryingRician
            # loss, snr = val_step(net, sents, sents, 0.18, pad_idx, criterion,
            #                      args.channel, seq_to_text)
            total += loss
            total_bce += bce_loss_val
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}; BCE: {bce_loss_val:.5f}')
    return total / len(val_iterator), total_bce / len(val_iterator)

def save_checkpoint(epoch, avg_loss, val_bce_loss, epoch_train_loss, train_bce_loss,
                    avg_mi_bits, snr_min, snr_max, snr_avg):
    checkpoint_path = os.path.join(
        args.checkpoint_path,
        f'checkpoint_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    )
    os.makedirs(args.checkpoint_path, exist_ok=True)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': ca_deepsc.state_dict(),
        # 'mi_net_state_dict': mi_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'mi_opt_state_dict': mi_opt.state_dict(),
        'loss': avg_loss,
        'val_bce_loss': val_bce_loss,
        'train_loss': epoch_train_loss,
        'train_bce_loss': train_bce_loss,
        'mi_bits': avg_mi_bits,
        'snr_min': snr_min,
        'snr_max': snr_max,
        'snr_avg': snr_avg,
    }, checkpoint_path)

    print(
        f"Checkpoint saved at {checkpoint_path} with epoch {epoch + 1}, val loss {avg_loss:.5f}, val BCE {val_bce_loss:.5f}, SNR Min: {snr_min:.2f}, Max: {snr_max:.2f}, Avg: {snr_avg:.2f}")

if __name__ == '__main__':
    # Check PyTorch's CUDA availability
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    signal.signal(signal.SIGINT, signal_handler)  # Bind Ctrl+C to stop training
    args = parser.parse_args()
    args.vocab_file = os.path.join('data',
                                   args.vocab_file)  # Simplified path joining
    # loss_file = os.path.join(args.checkpoint_path, 'losses.json')

    # Print the selected channel
    print(f"Selected Channel: {args.channel}")

    # Load vocabulary file
    with open(args.vocab_file, 'rb') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    seq_to_text = SeqtoText(token_to_idx, end_idx)

    "Test ham them nhieu vao data"

    # train_eur = EurDataset('train')

    # print("train_eur")

    # sample_sentence_1 = torch.tensor(train_eur[0][1], dtype=torch.long).unsqueeze(0).to(device)

    # print(sample_sentence_1)

    # StoT = SeqtoText(token_to_idx, end_idx)

    # input_text = StoT.sequence_to_text(sample_sentence_1.cpu().numpy().tolist()[0])

    # print(input_text)

    # train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
    #                             num_workers=0, pin_memory=True,
    #                             collate_fn=collate_pair_data)
    # pbar = tqdm(train_iterator)

    # for batch_idx, (noise_sents, trg, labels) in enumerate(train_iterator):
    #     for i in range(5):
    #         print(noise_sents[i])
    #         print(trg[i])
    #         print(labels[i])
    #     break


    "Bắt đầu từ đây"
    
    ca_deepsc = CA_DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)
    # mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    #Binary cross entropy
    bce_loss_fn = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(ca_deepsc.parameters(), lr=1e-4,
                                 betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # mi_opt = torch.optim.Adam(mi_net.parameters(), lr=0.001)

    initNetParams(ca_deepsc)

    # List available checkpoints
    list_checkpoints(args.checkpoint_path)

    # Prompt user for action with input validation
    # while True:
    #     action = input("Choose action: resume or start? ").strip().lower()
    #     if action in ['resume', 'start']:
    #         break
    #     print("Invalid input. Please enter 'resume' or 'start'.")

    action = args.action  # 'resume' hoặc 'start'

    start_epoch = 0

    if action == 'resume':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='latest')
        if checkpoint and checkpoint['epoch'] < args.epochs:
            start_epoch = checkpoint['epoch']
            ca_deepsc.load_state_dict(checkpoint['model_state_dict'])
            # mi_net.load_state_dict(checkpoint['mi_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # mi_opt.load_state_dict(checkpoint['mi_opt_state_dict'])
            print(
                f"Resuming from epoch {start_epoch} with loss {checkpoint['loss']:.5f}")
        else:
            print(
                "Cannot resume: Training completed or no valid checkpoint. Switching to 'start'.")
            
    if action == 'start':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='best')
        if checkpoint:
            ca_deepsc.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint['loss']
            print(f"Starting new phase with best model, loss {best_loss:.5f}")
        else:
            print("Starting from scratch: No best checkpoint found.")
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        # Training
        interrupted, epoch_train_loss, train_bce_loss, avg_mi_bits, snr_min, snr_max, snr_avg = train(
            epoch, args, ca_deepsc, num_vocab)
        if interrupted:
            print(
                f"Training stopped at epoch {epoch + 1}. Saving checkpoint...")
            avg_loss, val_bce_loss = validate(epoch, args, ca_deepsc, seq_to_text)
            save_checkpoint(epoch, avg_loss, val_bce_loss, epoch_train_loss, train_bce_loss,
                avg_mi_bits, snr_min, snr_max, snr_avg)
            break

        avg_loss, val_bce_loss = validate(epoch, args, ca_deepsc, seq_to_text)
        save_checkpoint(epoch, avg_loss, val_bce_loss, epoch_train_loss, train_bce_loss,
                avg_mi_bits, snr_min, snr_max, snr_avg)
        
        print(f"GPU Utilization: {torch.cuda.utilization(0)}%")
        print(
            f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB")

    # clean_checkpoints("checkpoints/deepsc-Rayleigh", keep_latest_n=5)

    print("Training finished.")