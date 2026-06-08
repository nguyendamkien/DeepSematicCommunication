import argparse
import json
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_pair_data
from models.transceiver import DeepSC
from performance import Similarity
from utils import SNR_to_noise, greedy_decode, SeqtoText, BleuScore, load_checkpoint, \
    debug_greedy_decode

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='train_data_with_error.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab_with_error.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='./kaggle/working/checkpoints/mask-fgm-deepsc-awgn',
                    type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--batch-size', default=1,
                    type=int)  # Set batch size to 1 for detailed observation
parser.add_argument('--SNR', default=18, type=int)  # Default SNR for testing
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--epochs', default=50, type=int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    args.vocab_file = './data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    print('test data')
    test_eur = EurDataset('test')
    print(len(test_eur))
    test_iterator = DataLoader(test_eur, 128,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_pair_data)
    seq_to_text = SeqtoText(token_to_idx, end_idx)
    for num_batch, (noise, clean) in enumerate(test_iterator):
        print(noise[0])
        print(clean[0])

        noise_text = seq_to_text.sequence_to_text(noise[0].cpu().numpy().tolist())
        clean_text = seq_to_text.sequence_to_text(clean[0].cpu().numpy().tolist())

        print(noise_text)
        print(clean_text)
        
        break