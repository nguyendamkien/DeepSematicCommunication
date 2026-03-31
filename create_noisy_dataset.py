import sys # interacting with the python system
from collections import Counter # count frequency of each element

import nltk # Natural Languages Toolkit - library for NLP
from matplotlib import pyplot as plt

import argparse # Cho phép truyền tham số khi chạy script
import json # read, write json data
import os 
import pickle # Save and load object Python (model, tokenizer, vocab)
import re # Handle string by regex
import unicodedata # Normalize Unicode
from tqdm import tqdm # Progress bar - thanh tien trinh
from w3lib.html import remove_tags # Remove HTML tags

from utils import add_semantic_noise, SeqtoText;
from dataset import EurDataset, collate_data

# Download NLTK data (run once if not already installed)
nltk.download('punkt', quiet=True)
# Argument parser for handling input and ouput direstories for text data processing
parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='europarl/txt/en', type=str)
parser.add_argument('--output-train-dir', default='train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='test_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab.json', type=str)
parser.add_argument('--output-noisy-test-sir', default='noisy_test_data.pkl', type=str)

# Special tokens used in the semantic communication model
SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

import random

def add_semantic_noise_sentence(src, vocab_size, prob=0.1, pad_idx=0):

    noise_types = ["substitute", "insert", "delete", "verb"]
    probs = [0.4, 0.2, 0.2, 0.2]

    new_sent = src.copy()

    for j in range(len(src)):

        # bỏ 4 token đặc biệt
        if src[j] <= 4:
            continue

        if random.random() < prob:

            noise_type = random.choices(noise_types, probs)[0]

            # -------- SUBSTITUTE (giữ nguyên) --------
            if noise_type == "substitute":
                new_sent[j] = random.randint(5, vocab_size - 1)

            # -------- DELETE (giữ nguyên logic dùng PAD) --------
            elif noise_type == "delete":
                new_sent[j] = pad_idx   # giữ đúng như bạn

            # -------- INSERT (fix nhẹ bug overwrite) --------
            elif noise_type == "insert":
                if j < len(src) - 1 and src[j+1] != pad_idx:
                    new_sent[j+1] = new_sent[j]   # giữ logic cũ
                    new_sent[j] = random.randint(5, vocab_size - 1)

            # -------- VERB (giữ nguyên range lớn) --------
            elif noise_type == "verb":
                delta = random.randint(-10, 10)
                new_token = src[j] + delta

                if 4 <= new_token < vocab_size:
                    new_sent[j] = new_token
                else:
                    new_sent[j] = random.randint(5, vocab_size - 1)

    return new_sent

if __name__ == '__main__':
    # Load vocabulary file
    args = parser.parse_args()
    args.vocab_file = os.path.join('data',
                                   args.vocab_file)
    with open(args.vocab_file, 'rb') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    print(num_vocab)
    vocab_size = len(vocab)
    print(num_vocab)

    noisy_test_data = []

    print("Loading test dataset...")
    test_eur = EurDataset('test')
    print(len(test_eur))

    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    StoT = SeqtoText(token_to_idx, end_idx)

    # print(test_eur.shape)

    # print(test_eur[0])

    # print(test_eur[1])
    # print(type(test_eur[0]))

    for src in tqdm(test_eur):

    #     print(src)

        noisy_src = add_semantic_noise_sentence(src, num_vocab, prob=0.1)

        noisy_test_data.append((noisy_src, src))

    print(noisy_test_data[:5])
    # lấy sample đầu

    for abs in noisy_test_data[:5]:
        noisy_src, original_src = abs

        print("Noisy:", noisy_src)
        print("Original:", original_src)
        print("Original:", StoT.sequence_to_text(original_src))
        print("Noisy:", StoT.sequence_to_text(noisy_src))

    args.output_noisy_test_sir = os.path.join('data',
                                   args.output_noisy_test_sir)
    with open(args.output_noisy_test_sir, 'wb') as f:
         pickle.dump(noisy_test_data, f)