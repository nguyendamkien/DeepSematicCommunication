#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_test_set.py
=====================
Tái sinh tập test từ câu gốc (clean) trong test_data_with_error.pkl.

Quy trình:
  1. Load test_data_with_error.pkl → trích xuất clean_ids
  2. Giải mã clean_ids → clean text (câu đúng) qua vocab
  3. Tái sinh lỗi với tỷ lệ --error-rate (0.0 = không lỗi, 1.0 = 100% token lỗi)
  4. Mã hoá lại và lưu pickle mới

Ví dụ chạy:
  python generate_test_set.py --error-rate 0.15 --output-test data/test_15pct_error.pkl
  python generate_test_set.py --error-rate 0.30 --output-test data/test_30pct_error.pkl
  python generate_test_set.py --error-rate 0.0  --output-test data/test_no_error.pkl
"""

import argparse
import json
import pickle
import random
import re
from collections import Counter

import nltk
from tqdm import tqdm

from utils import SeqtoText
from generation_error import (
    build_deletion_candidates,
    insert_error,
    delete_error,
    replace_error,
    verb_error,
)

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Regenerate test set from clean sentences with configurable error rate.'
)
parser.add_argument(
    '--input-test',
    default='data/test_data_with_error.pkl',
    type=str,
    help='Path to the original test_data_with_error.pkl file.'
)
parser.add_argument(
    '--vocab-file',
    default='data/vocab_with_error.json',
    type=str,
    help='Path to the vocab JSON file.'
)
parser.add_argument(
    '--output-test',
    default='data/0_percent_test_data_with_error.pkl',
    type=str,
    help='Output path for the new test pickle file.'
)
parser.add_argument(
    '--error-rate',
    default=0.4,
    type=float,
    help=(
        'Fraction of tokens per sentence that will be corrupted. '
        'Range: 0.0 (no errors) → 1.0 (all tokens corrupted). '
        'Default: 0.4 (40%).'
    )
)
parser.add_argument(
    '--error-type-probs',
    default='0.30,0.25,0.25,0.20',
    type=str,
    help=(
        'Comma-separated probabilities for error types: '
        'InsertError, VerbError, ReplaceError, DeleteError. '
        'Must sum to 1.0. Default: "0.30,0.25,0.25,0.20".'
    )
)
parser.add_argument(
    '--seed',
    default=42,
    type=int,
    help='Random seed for reproducibility. Default: 42.'
)


# ─────────────────────────────────────────────
# TOKENIZER  (Khớp với preprocess_text_with_error.py)
# ─────────────────────────────────────────────
def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')
    tokens = [t for t in s.split(delim) if t]
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


# ─────────────────────────────────────────────
# DECODE ids → text  (bỏ <START>, <END>, <PAD>)
# ─────────────────────────────────────────────
SKIP_TOKENS = {'<START>', '<END>', '<PAD>', '<UNK>'}

def ids_to_text(ids, idx_to_token):
    """Chuyển list token-id → chuỗi văn bản thô (bỏ special tokens & padding 0)."""
    words = []
    for idx in ids:
        if idx == 0:          # padding
            continue
        token = idx_to_token.get(str(idx)) or idx_to_token.get(idx)
        if token is None or token in SKIP_TOKENS:
            continue
        words.append(token)
    return ' '.join(words)


# ─────────────────────────────────────────────
# HÀM TẠO LỖI THEO TỶ LỆ (error_rate)
# ─────────────────────────────────────────────
def introduce_errors_by_rate(sentence, deletion_candidates, error_rate,
                              error_type_probs=(0.30, 0.25, 0.25, 0.20)):
    """
    Thêm lỗi vào câu dựa trên tỷ lệ token bị lỗi (error_rate).

    Với mỗi token trong câu, xác suất áp dụng 1 loại lỗi = error_rate.
    Sau khi áp dụng lỗi, nhãn (label) 0/1 được cập nhật tương ứng.

    Các loại lỗi:
      0 = InsertError  – chèn từ thừa
      1 = VerbError    – chia động từ sai
      2 = ReplaceError – thay từ
      3 = DeleteError  – xóa từ

    Returns:
        (noisy_sentence: str, labels: list[int])
    """
    if error_rate <= 0.0:
        words = sentence.split()
        labels = [0] * len(words)
        return sentence, labels

    words = sentence.split()
    if not words:
        return sentence, []

    labels = [0] * len(words)

    # Số token dự kiến bị lỗi
    n_errors = max(1, round(len(words) * error_rate)) if error_rate > 0 else 0

    # Chọn ngẫu nhiên vị trí sẽ bị lỗi (không trùng)
    n_errors = min(n_errors, len(words))
    error_positions = sorted(
        random.sample(range(len(words)), n_errors), reverse=True
    )
    # Dùng reverse để tránh index dịch khi insert/delete

    for pos in error_positions:
        if len(words) == 0:
            break

        # Đảm bảo pos hợp lệ sau các lần chỉnh sửa trước
        pos = min(pos, len(words) - 1)

        # Chọn loại lỗi
        error_type = random.choices(
            range(4), weights=list(error_type_probs)
        )[0]

        if error_type == 0:
            # InsertError: chèn từ thừa vào trước pos
            words = insert_error(words, pos, deletion_candidates)
            labels.insert(pos, 1)

        elif error_type == 1:
            # VerbError: chia động từ sai tại pos nếu là verb
            # Thử trên sub-list để tránh tagging lại toàn bộ câu
            words, changed_idx = verb_error(words)
            if changed_idx is not None:
                # Ensure label list stays in sync
                while len(labels) < len(words):
                    labels.append(0)
                labels[changed_idx] = 1

        elif error_type == 2:
            # ReplaceError
            words = replace_error(words, pos, deletion_candidates)
            while len(labels) < len(words):
                labels.append(0)
            labels[pos] = 1

        elif error_type == 3:
            # DeleteError: xóa từ tại pos
            if len(words) > 1:
                words.pop(pos)
                labels.pop(pos)

        # Đồng bộ độ dài labels và words
        while len(labels) < len(words):
            labels.append(0)
        labels = labels[:len(words)]

    return ' '.join(words), labels


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(args):
    random.seed(args.seed)

    # Parse error_type_probs
    try:
        et_probs = [float(x) for x in args.error_type_probs.split(',')]
        assert len(et_probs) == 4, "Cần đúng 4 xác suất."
        s = sum(et_probs)
        et_probs = [p / s for p in et_probs]  # chuẩn hoá tổng = 1
    except Exception as e:
        print(f"[ERROR] --error-type-probs không hợp lệ: {e}")
        return

    print(f"[CONFIG] error_rate      = {args.error_rate:.2%}")
    print(f"[CONFIG] error_type_probs= {et_probs}")
    print(f"[CONFIG] input_test      = {args.input_test}")
    print(f"[CONFIG] vocab_file      = {args.vocab_file}")
    print(f"[CONFIG] output_test     = {args.output_test}")
    print(f"[CONFIG] seed            = {args.seed}")
    print()

    # ── 1. Load vocab ──────────────────────────────────────────────────────
    print("Loading vocab...")
    with open(args.vocab_file, 'r') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    # Hỗ trợ cả int key và str key khi tra cứu
    idx_to_token = {str(v): k for k, v in token_to_idx.items()}
    print(f"  Vocab size: {len(token_to_idx)}")

    # ── 2. Load test data ──────────────────────────────────────────────────
    print(f"Loading test data from: {args.input_test}")
    with open(args.input_test, 'rb') as f:
        test_data = pickle.load(f)
    print(f"  Total samples: {len(test_data)}")
    print(f"  Test data: {test_data[0]}")

    # ── 3. Trích xuất clean sentences ─────────────────────────────────────
    print("Extracting clean sentences from test data...")
    clean_sentences = []
    for noise_ids, clean_ids, labels in test_data:
        clean_text = ids_to_text(clean_ids, idx_to_token)
        if clean_text.strip():
            clean_sentences.append(clean_text)

    print(f"  Clean sentences: {clean_sentences[:5]}")

    # Bỏ trùng để xây deletion_candidates tốt hơn
    unique_clean = list(set(clean_sentences))
    print(f"  Unique clean sentences: {len(unique_clean)}")

    # ── 4. Xây deletion_candidates từ clean corpus ─────────────────────────
    print("Building deletion candidates...")
    deletion_candidates = build_deletion_candidates(unique_clean)

    # # ── 5. Tái sinh lỗi ───────────────────────────────────────────────────
    print(f"Generating noisy sentences (error_rate={args.error_rate:.2%})...")
    results = []
    error_token_counts = []
    total_token_counts = []

    for clean_text in tqdm(clean_sentences, desc="Injecting errors"):
        noisy_text, labels = introduce_errors_by_rate(
            clean_text,
            deletion_candidates,
            error_rate=args.error_rate,
            error_type_probs=tuple(et_probs),
        )

        # Thống kê tỷ lệ lỗi thực tế
        error_token_counts.append(sum(labels))
        total_token_counts.append(len(labels))

        # Encode lại thành token ids
        noisy_tokens = tokenize(
            noisy_text,
            punct_to_keep=[';', ','],
            punct_to_remove=['?', '.']
        )
        clean_tokens = tokenize(
            clean_text,
            punct_to_keep=[';', ','],
            punct_to_remove=['?', '.']
        )

        # Thêm label 0 cho <START> và <END>
        full_labels = [0] + labels + [0]

        noise_ids = [token_to_idx.get(w, token_to_idx.get('<UNK>', 3))
                     for w in noisy_tokens]
        clean_ids = [token_to_idx.get(w, token_to_idx.get('<UNK>', 3))
                     for w in clean_tokens]

        results.append((noise_ids, clean_ids, full_labels))

    # # ── 6. Thống kê ───────────────────────────────────────────────────────
    total_err = sum(error_token_counts)
    total_tok = sum(total_token_counts)
    actual_rate = total_err / total_tok if total_tok > 0 else 0.0
    print(f"\n[STATS] Target error rate : {args.error_rate:.2%}")
    print(f"[STATS] Actual error rate : {actual_rate:.2%}  "
          f"({total_err}/{total_tok} tokens)")

    # # Mẫu
    print("\n[SAMPLE] First 3 samples:")
    for i, (n_ids, c_ids, lbls) in enumerate(results[:3]):
        n_txt = ids_to_text(n_ids, idx_to_token)
        c_txt = ids_to_text(c_ids, idx_to_token)
        print(f"  [{i}] CLEAN : {c_txt}")
        print(f"  [{i}] NOISY : {n_txt}")
        print(f"  [{i}] LABELS: {lbls}")
        print()

    # ── 7. Lưu file ───────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.output_test) if os.path.dirname(args.output_test) else '.', exist_ok=True)
    with open(args.output_test, 'wb') as f:
        pickle.dump(results, f)
    print(f"[SAVED] {len(results)} samples → {args.output_test}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
