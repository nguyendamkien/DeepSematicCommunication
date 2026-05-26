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
from models.transceiver import DeepSC
from utils import SNR_to_noise, train_step, train_mask, val_step, initNetParams, \
    SeqtoText, list_checkpoints, load_checkpoint

plt.ion() # Turn on interactive mode

# Argument parser for configuring hyperparameters and paths
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='vocab_with_error.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='/kaggle/working/checkpoints/mask-deepsc-AWGN',
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
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--epsilon', default=0.5, type=float)
parser.add_argument('--lrate', default=1e-4, type=float)
parser.add_argument('--weightdecay', default=1e-4, type=float)
parser.add_argument('--lamdaAdv', default=0.5, type=float)

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
def train(epoch, args, net):
    global stop_training
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                num_workers=4, pin_memory=True,
                                collate_fn=collate_pair_data)
    pbar = tqdm(train_iterator)
    # For TimeVaryingRician
    # noise_std_options = np.arange(0.045, 0.316, 0.010)
    epoch_loss = 0
    loss_adv_total = 0
    mask_loss = 0
    batch_count = 0
    snr_values = []

    for noise_sents, clean_sents in pbar:
        if stop_training:
            return True, epoch_loss, loss_adv_total, mask_loss, min(
                snr_values) if snr_values else 0, max(
                snr_values) if snr_values else 0, sum(snr_values) / len(
                snr_values) if snr_values else 0
        noise_sents = noise_sents.to(device)
        clean_sents = clean_sents.to(device)
        noise_std = float(
            np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0])
        loss_total, loss_adv, snr = train_step(net, noise_sents, clean_sents, noise_std, pad_idx,
                                     optimizer_deepsc, criterion, args.channel, args.epsilon, args.lamdaAdv)
        loss_mask, snr = train_mask(net, noise_sents, clean_sents, noise_std, pad_idx, optimizer_mask, criterion, args.channel)
        epoch_loss += loss_total
        loss_adv_total += loss_adv
        mask_loss += loss_mask
        batch_count += 1
        snr_values.append(snr)
        avg_mask_loss = mask_loss / batch_count
        pbar.set_description(
            f'Epoch: {epoch + 1}; Type: Train; Loss: {loss_total:.5f}; Loss_adv:{loss_adv:.5f}; Loss_mask: {avg_mask_loss:.5f}; SNR: {snr:.5f}; Noise Std: {noise_std:.5f}')

    snr_min = min(snr_values) if snr_values else 0
    snr_max = max(snr_values) if snr_values else 0
    snr_avg = sum(snr_values) / len(snr_values) if snr_values else 0

    avg_epoch_loss = epoch_loss / len(train_iterator)
    avg_adv_loss = loss_adv_total / len(train_iterator)
    avg_mask_loss = mask_loss / len(train_iterator)
    return False, avg_epoch_loss, avg_adv_loss, avg_mask_loss, snr_min, snr_max, snr_avg

# Validation function
def validate(epoch, args, net, seq_to_text):
    val_eur = EurDataset('val')  # Load test dataset
    val_iterator = DataLoader(val_eur, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_pair_data)
    net.eval()
    pbar = tqdm(val_iterator)
    total = 0
    with torch.no_grad():
        for noise_sents, clean_sents in pbar:
            # print(f"Batch contains {sents.shape[0]} sentences")
            noise_sents = noise_sents.to(device)
            clean_sents = clean_sents.to(device)
            loss, snr = val_step(net, noise_sents, clean_sents, 0.1, pad_idx, criterion,
                                 args.channel, seq_to_text)
            total += loss
            pbar.set_description(
                f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')
    return total / len(val_iterator)

# Function to save checkpoint for each epoch
def save_checkpoint(epoch, avg_loss, epoch_train_loss,
                    avg_adv_loss, mask_train_loss,
                    snr_min, snr_max, snr_avg):
    checkpoint_path = os.path.join(
        args.checkpoint_path,
        f'checkpoint_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
    )
    os.makedirs(args.checkpoint_path, exist_ok=True)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': deepsc.state_dict(),
        'optimizer_deepsc_state_dict': optimizer_deepsc.state_dict(),
        'optimizer_mask_state_dict': optimizer_mask.state_dict(),
        'loss': avg_loss,
        'train_loss': epoch_train_loss,
        'loss_adv': avg_adv_loss,
        'mask_train_loss': mask_train_loss,
        'snr_min': snr_min,
        'snr_max': snr_max,
        'snr_avg': snr_avg,
    }, checkpoint_path)

    print(
        f"Checkpoint saved at {checkpoint_path} with epoch {epoch + 1}, validation loss {avg_loss:.5f}, SNR Min: {snr_min:.2f}, Max: {snr_max:.2f}, Avg: {snr_avg:.2f}")

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
    
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Collect mask parameters (mask_perturbation_model only)
    # calibration (ACN) is intentionally excluded — trained by optimizer_deepsc in train_step
    mask_params = []
    for layer in deepsc.decoder.dec_layers:
        mask_params.extend(list(layer.src_mha.mask_perturbation_model.parameters()))

    # Parameters for DeepSC excluding mask — includes calibration (ACN)
    mask_param_ids = set(id(p) for p in mask_params)
    deepsc_params = [p for p in deepsc.parameters() if id(p) not in mask_param_ids]
    
    optimizer_deepsc = torch.optim.Adam(deepsc_params, lr=args.lrate,
                                 betas=(0.9, 0.98), eps=1e-8, weight_decay=args.weightdecay)
    optimizer_mask = torch.optim.Adam(mask_params, lr=args.lrate,
                                 betas=(0.9, 0.98), eps=1e-8, weight_decay=args.weightdecay)

    initNetParams(deepsc)

    # List available checkpoints
    list_checkpoints(args.checkpoint_path)

    action = args.action  # 'resume' hoặc 'start'

    start_epoch = 0

    if action == 'resume':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='latest')
        if checkpoint and checkpoint['epoch'] < args.epochs:
            start_epoch = checkpoint['epoch']
            deepsc.load_state_dict(checkpoint['model_state_dict'])
            optimizer_deepsc.load_state_dict(checkpoint['optimizer_deepsc_state_dict'])
            optimizer_mask.load_state_dict(checkpoint['optimizer_mask_state_dict'])
            print(
                f"Resuming from epoch {start_epoch} with loss {checkpoint['loss']:.5f}")
        else:
            print(
                "Cannot resume: Training completed or no valid checkpoint. Switching to 'start'.")
            
    if action == 'start':
        checkpoint = load_checkpoint(args.checkpoint_path, mode='best')
        if checkpoint:
            deepsc.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint['loss']
            print(f"Starting new phase with best model, loss {best_loss:.5f}")
        else:
            print("Starting from scratch: No best checkpoint found.")
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        # Training
        interrupted, epoch_train_loss, avg_adv_loss, mask_train_loss, snr_min, snr_max, snr_avg = train(
            epoch, args, deepsc)
        if interrupted:
            print(
                f"Training stopped at epoch {epoch + 1}. Saving checkpoint...")
            avg_loss = validate(epoch, args, deepsc, seq_to_text)
            save_checkpoint(epoch, avg_loss, epoch_train_loss, avg_adv_loss, mask_train_loss,
                snr_min, snr_max, snr_avg)
            break

        avg_loss = validate(epoch, args, deepsc, seq_to_text)
        save_checkpoint(epoch, avg_loss, epoch_train_loss, avg_adv_loss,
                mask_train_loss, snr_min, snr_max, snr_avg)
        
        print(f"GPU Utilization: {torch.cuda.utilization(0)}%")
        print(
            f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB")

    print("Training finished.")

