# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie (Adapted)
@File: inference.py
@Time: Updated on 2025/1/23
"""
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
from models.transceiver_calibration import CA_DeepSC
from performance import Similarity
from utils import SNR_to_noise, greedy_decode, SeqtoText, BleuScore, load_checkpoint, \
    debug_greedy_decode, greedy_decode_calibration

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab.json', type=str)
# parser.add_argument('--checkpoint-path',
#                     default='/kaggle/working/checkpoints/deepsc-Rayleigh',
#                     type=str)
# parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--checkpoint-path',
                    default='./kaggle/working/checkpoints/ca-deepsc-AWGN',
                    type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--batch-size', default=1,
                    type=int)  # Set batch size to 1 for detailed observation
parser.add_argument('--SNR', default=12, type=int)  # Default SNR for testing
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--epochs', default=50, type=int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(args, snr, net):
    # Initialize metrics calculators
    similarity = Similarity(
        batch_size=1)  # Small batch size for single sentences
    bleu_score_calc = BleuScore(1, 0, 0, 0)  # 1-gram BLEU score

    # Load the test dataset
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size,
                               num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    
    # Initialize sequence-to-text converter using vocabulary
    StoT = SeqtoText(token_to_idx, end_idx)

    net.eval()
    results = []  # To store input-output pairs for observation
    noise_std = SNR_to_noise(snr)

    print(f"Starting inference with SNR: {snr}...")

    with torch.no_grad():
        for batch_index, sents in enumerate(
                tqdm(test_iterator, desc="Inference Progress"), start=1):
            # Move data to the appropriate device
            sents = sents.to(device)

            # Decode sentences
            out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                start_idx, args.channel)

            # Convert input (target) and model output to text
            input_sentences = sents.cpu().numpy().tolist()
            decoded_sentences = out.cpu().numpy().tolist()

            input_texts = list(map(StoT.sequence_to_text, input_sentences))
            output_texts = list(map(StoT.sequence_to_text, decoded_sentences))

            # Calculate metrics
            bleu = bleu_score_calc.compute_blue_score(input_texts, output_texts)
            sim = similarity.compute_similarity(input_texts, output_texts)

            # Save input-output pairs with metrics for observation
            for i, (input_text, output_text) in enumerate(
                    zip(input_texts, output_texts)):
                results.append({
                    "Input": input_text,
                    "Output": output_text,
                    "BLEU": bleu[i] if isinstance(bleu, list) else bleu,
                    "Similarity": sim[i] if isinstance(sim, list) else sim
                })
                print(f"\nInput: {input_text}")
                print(f"Output: {output_text}")
                print(
                    f"BLEU Score: {bleu[i] if isinstance(bleu, list) else bleu:.4f}")
                print(
                    f"Similarity Score: {sim[i] if isinstance(sim, list) else sim:.4f}\n")

    print("Inference completed successfully.")
    return results

def debug_similarity(similarity_calculator: Similarity, sent1: str,
                     sent2: str) -> None:
    """Debug similarity calculation between two sentences"""
    print("\nSimilarity Debugging:")
    print(f"Sentence 1: {sent1}")
    print(f"Sentence 2: {sent2}")

    # Get embeddings
    with torch.no_grad():
        embeddings1 = similarity_calculator.get_sentence_embeddings([sent1])
        embeddings2 = similarity_calculator.get_sentence_embeddings([sent2])

        # Print embedding information
        print(f"Embedding shapes: {embeddings1.shape}, {embeddings2.shape}")
        print(
            f"Embedding norms: {torch.norm(embeddings1):.4f}, {torch.norm(embeddings2):.4f}")

        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings1,
                                                           embeddings2)
        print(f"Calculated similarity: {similarity.item():.4f}")

def sample_dataset_sentences(dataset: EurDataset, num_samples: int,
                             batch_size: int) -> DataLoader:
    """Sample sentences from dataset while ensuring they are batched and padded correctly."""
    sampled_indices = random.sample(range(len(dataset)),
                                    min(num_samples, len(dataset)))
    sampled_subset = torch.utils.data.Subset(dataset, sampled_indices)

    return DataLoader(
        sampled_subset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data  # Ensures padding is applied
    )


def test_sample_sentences(args, net, dataloader: DataLoader,
                          similarity_calculator: Similarity) -> None:
    """Test model on a batch of sampled and padded sentences, using debug mode for single sample"""
    print(f"\nTesting {len(dataloader.dataset)} sample sentences from dataset:")
    print(f"Current SNR: {args.SNR} dB")

    StoT = SeqtoText(token_to_idx, end_idx)
    bleu_score_calc = BleuScore(1, 0, 0, 0)

    for batch_idx, input_batch in enumerate(dataloader):
        if isinstance(input_batch, torch.Tensor):
            input_batch = input_batch.to(device)
        else:
            input_batch = input_batch[0].to(device)

        with torch.no_grad():
            noise_std = SNR_to_noise(args.SNR)
            # Use debug_greedy_decode for single sample, regular greedy_decode otherwise
            if len(dataloader.dataset) == 1:
                output_tokens = debug_greedy_decode(
                    net, input_batch, noise_std, args.MAX_LENGTH, pad_idx,
                    start_idx,
                    args.channel, seq_to_text=StoT
                )
            else:
                output_tokens, snr = greedy_decode(
                    net, input_batch, noise_std, args.MAX_LENGTH, pad_idx,
                    start_idx,
                    args.channel,
                    device
                )

        for i, output_tensor in enumerate(output_tokens):
            output_sentence = StoT.sequence_to_text(
                output_tensor.cpu().numpy().tolist())
            original_sentence = StoT.sequence_to_text(
                input_batch[i].cpu().numpy().tolist())
            bleu = bleu_score_calc.compute_blue_score([original_sentence],
                                                      [output_sentence])[0]
            sim = similarity_calculator.compute_similarity([original_sentence],
                                                           [output_sentence])[0]

            print(
                f"\nTest {batch_idx * args.batch_size + i + 1}/{len(dataloader.dataset)}:")
            print(f"Input: {original_sentence}")
            print(f"Output: {output_sentence}")
            print(f"BLEU Score: {bleu:.4f}")
            print(f"Similarity Score: {sim:.4f}")


def interactive_test(args, snr, net):
    """Interactive test mode with similarity debugging"""
    similarity = Similarity(batch_size=1)
    bleu_score_calc = BleuScore(1, 0, 0, 0)
    StoT = SeqtoText(token_to_idx, end_idx)

    print("\nAvailable commands:")
    print("- Enter a sentence to test")
    print(
        "- Type 'sample N' to test N random sentences from dataset (e.g., 'sample 5')")
    print("- Type 'terminate' to exit")
    print(f"\nCurrent SNR: {snr} dB")

    while True:
        try:
            user_input = input("\nInput: ").strip()

            # Check for commands
            if user_input.lower() == "terminate":
                print("Exiting interactive test.")
                break

            elif user_input.lower().startswith("sample"):
                try:
                    num_samples = int(user_input.split()[1])
                    test_dataset = EurDataset('test')
                    # Use DataLoader instead of manually sampling
                    sampled_dataloader = sample_dataset_sentences(test_dataset,
                                                                  num_samples,
                                                                  args.batch_size)

                    test_sample_sentences(args, net, sampled_dataloader,
                                          similarity)
                    continue
                except Exception as e:  # Catch all exceptions and print the actual error
                    print(f"Error: {e}")
                    continue

            # Process single sentence
            input_tokens = [start_idx] + [
                token_to_idx.get(word, token_to_idx["<UNK>"])
                for word in user_input.split()
            ] + [4] + [end_idx]

            if len(input_tokens) > args.MAX_LENGTH + 2:  # +2 for start/end tokens
                print(
                    f"Input too long. Maximum length is {args.MAX_LENGTH} words.")
                continue

            input_tensor = torch.tensor(input_tokens,
                                        dtype=torch.long).unsqueeze(0).to(
                device)
            
            # Perform inference
            with torch.no_grad():
                noise_std = SNR_to_noise(args.SNR)
                # output_tokens = greedy_decode(net, input_tensor, noise_std,
                #                               args.MAX_LENGTH, pad_idx,
                #                               start_idx,
                #                               args.channel,
                #                               device)
                decoded, _ = greedy_decode_calibration(
                    net, input_tensor, noise_std,
                    args.MAX_LENGTH, pad_idx, start_idx,
                    args.channel, device
                )
                output_tokens = decoded
                print(input_tensor)
                print(output_tokens)

            # Process output tokens
            if isinstance(output_tokens, torch.Tensor):
                output_tokens = output_tokens.squeeze(0).cpu().numpy().tolist()

            # Loại bỏ token đặc biệt và cắt sau <END>
            clean_tokens = []
            for t in output_tokens:
                if t == end_idx:  # token <END>
                    break          # dừng lấy token sau <END>
                if t in (start_idx, 4):  # bỏ <START> và token 4
                    continue
                clean_tokens.append(t)

            # Chuyển sang text
            output_sentence = StoT.sequence_to_text(clean_tokens)

            # Convert output tokens back to text
            # output_sentence = StoT.sequence_to_text(output_tokens)

            print(user_input)
            print(output_sentence)

            # Calculate metrics
            bleu = \
                bleu_score_calc.compute_blue_score([user_input],
                                                   [output_sentence])[
                    0]
            sim = similarity.compute_similarity([user_input],
                                                [output_sentence])[0]

            # Print results
            print(f"\nResults:")
            print(f"Output: {output_sentence}")
            print(f"BLEU Score: {bleu:.4f}")
            print(f"Similarity Score: {sim:.4f}")

            # Helper to convert tensors/arrays/lists to scalars safely
            def to_scalar(x):
                # Torch tensors
                if isinstance(x, torch.Tensor):
                    if x.numel() == 1:
                        return x.item()
                    return float(x.mean().item())
                # NumPy arrays / lists / tuples
                if isinstance(x, (list, tuple, np.ndarray)):
                    arr = np.array(x)
                    if arr.size == 1:
                        return float(arr.item())
                    return float(arr.mean())
                # Floats/ints
                try:
                    return float(x)
                except Exception:
                    return 0.0

            # Debug similarity if score seems unexpected (use scalars to avoid ambiguous tensor comparison)
            if to_scalar(sim) > 0.8 and to_scalar(bleu) < 0.2:
                debug_similarity(similarity, user_input, output_sentence)

        except KeyboardInterrupt:
            print("\nExiting interactive test.")
            break
        except Exception as e:
            print(f"Error processing input: {str(e)}")


if __name__ == '__main__':
    # Check PyTorch's CUDA availability
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    args = parser.parse_args()

    # SNR value for inference
    SNR = args.SNR

    # Load vocabulary
    # args.vocab_file = '/kaggle/input/deepsc/data/' + args.vocab_file
    args.vocab_file = './data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # Define and load the model
    deepsc = CA_DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                    args.d_model, args.num_heads, args.dff, 0.1).to(device)

    checkpoint = load_checkpoint(args.checkpoint_path, mode='best')
    if checkpoint:
        deepsc.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['loss']
        print(f"Loaded best checkpoint with loss {best_loss:.5f}")
    else:
        print("No best checkpoint found.")
    # Verify that sentences are correctly padded and that the source mask is properly generated.
    test_eur = EurDataset('noisy_test')  # Load test dataset
    test_iterator = DataLoader(test_eur, 128,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_pair_data)
    seq_to_text = SeqtoText(token_to_idx, end_idx)
    # --- Test một câu từ test set để BLEU cao ---
    test_dataset = EurDataset('noisy_test')
    print(test_dataset[0][0])
    # Chọn câu đầu tiên từ test set, convert sang tensor
    sample_sentence_1 = torch.tensor(test_dataset[4][0], dtype=torch.long).unsqueeze(0).to(device)
    sample_sentence_2 = torch.tensor(test_dataset[4][1], dtype=torch.long).unsqueeze(0).to(device)
    StoT = SeqtoText(token_to_idx, end_idx)
    bleu_score_calc = BleuScore(1, 0, 0, 0)
    similarity = Similarity(batch_size=1)

    with torch.no_grad():
        noise_std = SNR_to_noise(SNR)
        output_tokens, _ = greedy_decode_calibration(deepsc, sample_sentence_1, noise_std,
                                    args.MAX_LENGTH, pad_idx, start_idx, args.channel, device)

    # Convert tokens to text
    input_text = StoT.sequence_to_text(sample_sentence_2.cpu().numpy().tolist()[0])
    sentence_1 = output_tokens.cpu().numpy().tolist()[0]
    output_text = StoT.sequence_to_text(sentence_1)
    print(sample_sentence_1)
    print(output_tokens)

    bleu = bleu_score_calc.compute_blue_score([input_text], [output_text])[0]
    sim = similarity.compute_similarity([input_text], [output_text])[0]

    print("\n--- Test hardcoded sample from test set ---")
    print(f"Input : {input_text}")
    print(f"Output: {output_text}")
    print(f"BLEU  : {bleu:.4f}")
    print(f"Sim   : {sim:.4f}\n")
    # ---------------------------------------------


    interactive_test(args, SNR, deepsc)
    #list_checkpoints("D:/timevaryingrician_checkpoints")
    #list_checkpoints("./kaggle/working/checkpoints/deepsc-AWGN")

