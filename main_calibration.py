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
    SeqtoText, list_checkpoints, load_checkpoint

plt.ion() # Turn on interactive mode

# Argument parser for configuring hyperparameters and paths
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='vocab_with_error.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='/kaggle/working/checkpoints/ca-adv-deepsc-AWGN',
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
def train(epoch, args, net, mi_net=None):
    global stop_training
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                num_workers=0, pin_memory=True,
                                collate_fn=collate_pair_data)
    pbar = tqdm(train_iterator)
    # For TimeVaryingRician
    # noise_std_options = np.arange(0.045, 0.316, 0.010)
    epoch_loss_total = 0
    epoch_loss_clean = 0
    epoch_loss_adv = 0
    mi_bits_total = 0
    batch_count = 0
    snr_values = []

    #noise_sent la cau goc, clean la target muon nhan
    for noise_sents, clean_sents, labels in pbar:
        if stop_training:
            return True, epoch_loss_total, epoch_loss_clean, epoch_loss_adv, mi_bits_total / batch_count if batch_count > 0 else 0, min(
                snr_values) if snr_values else 0, max(
                snr_values) if snr_values else 0, sum(snr_values) / len(
                snr_values) if snr_values else 0
        noise_sents = noise_sents.to(device)
        clean_sents = clean_sents.to(device)
        labels = labels.to(device)
        # noise_std = np.random.choice(noise_std_options, size=1).item()  # Scalar
        # For original Channel
        noise_std = float(
            np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0])
        # if mi_net is not None:
        #     mi_loss, mi_bits = train_mi(net, mi_net, sents, 0.1, pad_idx,
        #                                 mi_opt, args.channel)
        #     loss_total, snr = train_step(net, sents, sents, 0.1, pad_idx,
        #                                  optimizer, criterion, args.channel,
        #                                  mi_net)
        #     epoch_loss += loss_total
        #     mi_bits_total += mi_bits
        #     batch_count += 1
        #     snr_values.append(snr)
        #     pbar.set_description(
        #         f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; MI Loss: {mi_loss:.5f}; MI (bits): {mi_bits:.5f}; SNR: {snr:.5f}')
        # else:
        loss_total, loss_clean, loss_adv, snr = train_step_calibration(net, noise_sents, clean_sents, labels, noise_std, pad_idx,
                                     optimizer, criterion, args.channel, bce_loss_fn)
        epoch_loss_total += loss_total
        epoch_loss_clean += loss_clean
        epoch_loss_adv += loss_adv
        snr_values.append(snr)
        pbar.set_description(
            f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; SNR: {snr:.5f}; Noise Std: {noise_std:.5f}')

    snr_min = min(snr_values) if snr_values else 0
    snr_max = max(snr_values) if snr_values else 0
    snr_avg = sum(snr_values) / len(snr_values) if snr_values else 0

    avg_epoch_loss = epoch_loss_total / len(train_iterator)
    avg_epoch_loss_clean = epoch_loss_clean / len(train_iterator)
    avg_epoch_loss_adv = epoch_loss_adv / len(train_iterator)
    avg_mi_bits = mi_bits_total / batch_count if batch_count > 0 else 0
    return False, avg_epoch_loss, avg_epoch_loss_clean, avg_epoch_loss_adv, avg_mi_bits, snr_min, snr_max, snr_avg

# Validation function
def validate(epoch, args, net, seq_to_text):
    val_eur = EurDataset('val')  # Load test dataset
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
    # Noise_std for TimeVaryingRician
    # noise_std_options = np.arange(0.045, 0.316, 0.010)
    # noise_std = np.random.choice(noise_std_options, size=1)
    with torch.no_grad():
        for noise_sents, clean_sents, labels in pbar:
            # print(f"Batch contains {sents.shape[0]} sentences")
            noise_sents = noise_sents.to(device)
            clean_sents = clean_sents.to(device)
            labels = labels.to(device)
            loss, snr = val_step_calibration(net, noise_sents, clean_sents, labels, 0.1, pad_idx, criterion,
                                 args.channel, bce_loss_fn)
            # TimeVaryingRician
            # loss, snr = val_step(net, sents, sents, 0.18, pad_idx, criterion,
            #                      args.channel, seq_to_text)
            total += loss
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')
    return total / len(val_iterator)

def save_checkpoint(epoch, avg_loss, epoch_train_loss_total, epoch_train_loss_clean, epoch_train_loss_adv,
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
        'scheduler_state_dict': scheduler.state_dict(),  # Lưu scheduler state
        # 'mi_opt_state_dict': mi_opt.state_dict(),
        'loss': avg_loss,
        'train_loss_total': epoch_train_loss_total,
        'train_loss_clean': epoch_train_loss_clean,
        'train_loss_adv': epoch_train_loss_adv,
        'mi_bits': avg_mi_bits,
        'snr_min': snr_min,
        'snr_max': snr_max,
        'snr_avg': snr_avg,
    }, checkpoint_path)

    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Checkpoint saved at {checkpoint_path} with epoch {epoch + 1}, val loss {avg_loss:.5f}, "
        f"LR: {current_lr:.2e}, SNR Min: {snr_min:.2f}, Max: {snr_max:.2f}, Avg: {snr_avg:.2f}")

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
    # LR Scheduler: giảm LR x0.5 nếu val loss không cải thiện sau 2 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6,
        verbose=True
    )
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
            # Restore scheduler state nếu có (backward-compatible)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        interrupted, epoch_train_loss_total, epoch_train_loss_clean, epoch_train_loss_adv, avg_mi_bits, snr_min, snr_max, snr_avg = train(
            epoch, args, ca_deepsc)
        if interrupted:
            print(
                f"Training stopped at epoch {epoch + 1}. Saving checkpoint...")
            avg_loss = validate(epoch, args, ca_deepsc, seq_to_text)
            save_checkpoint(epoch, avg_loss, epoch_train_loss_total, epoch_train_loss_clean, epoch_train_loss_adv,
                avg_mi_bits, snr_min, snr_max, snr_avg)
            break

        avg_loss = validate(epoch, args, ca_deepsc, seq_to_text)
        save_checkpoint(epoch, avg_loss, epoch_train_loss_total, epoch_train_loss_clean, epoch_train_loss_adv,
                avg_mi_bits, snr_min, snr_max, snr_avg)

        # Bước scheduler dựa trên val loss để tự động giảm LR khi cần
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.2e}")

        print(f"GPU Utilization: {torch.cuda.utilization(0)}%")
        print(
            f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB")

    # clean_checkpoints("checkpoints/deepsc-Rayleigh", keep_latest_n=5)

    print("Training finished.")

