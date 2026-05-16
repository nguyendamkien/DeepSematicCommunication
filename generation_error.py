import random
import re
from collections import Counter
from tqdm import tqdm

from lemminflect import getInflection
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# =========================
# MULTINOULLI SAMPLING
# =========================
def multinoulli(probs):
    """Lấy mẫu theo phân phối multinoulli."""
    return random.choices(range(len(probs)), weights=probs)[0]


# =========================
# BUILD DELETION CANDIDATES
# Danh sách các từ "thường bị xóa" — dùng cho InsertError & ReplaceError.
# Đây là những từ chức năng/phụ trợ phổ biến, dễ bị thêm thừa.
# =========================
COMMON_DELETION_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "that", "this", "these", "those",
    "not", "no", "very", "just", "also", "then", "than", "so", "but",
    "and", "or", "if", "as", "when", "where", "which", "who"
]


def build_deletion_candidates(sentences, top_k=500):
    """
    Xây dựng danh sách các từ thường bị xóa từ corpus.
    Ưu tiên dùng COMMON_DELETION_WORDS, bổ sung bằng các từ phổ biến nhất
    trong corpus (top_k từ có tần suất cao thường là từ chức năng).
    """
    counter = Counter()
    for sent in sentences:
        words = re.findall(r"\w+", sent.lower())
        counter.update(words)
    corpus_common = [w for w, _ in counter.most_common(top_k)]

    # Gộp: ưu tiên danh sách cứng, bổ sung từ corpus
    combined = list(COMMON_DELETION_WORDS)
    for w in corpus_common:
        if w not in combined:
            combined.append(w)
    return combined

# Các dạng verb chuẩn theo POS tag
VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBZ"]

lemmatizer = WordNetLemmatizer()

# ------------------------
# 1. Lemmatize đúng - đưa động từ về dạng gốc
# ------------------------
def get_lemma(word, pos_tag):
    if pos_tag.startswith("VB"):
        return lemmatizer.lemmatize(word, 'v')
    return word


# ------------------------
# 2. Inflect từ lemma - chuyển đổi dạng của động từ
# ------------------------
def inflect_verb(word, pos_tag):
    lemma = get_lemma(word, pos_tag)

    forms = set()
    for tag in VERB_TAGS:
        infl = getInflection(lemma, tag=tag)
        if infl:
            forms.update(infl)

    # bỏ dạng gốc
    forms = [f for f in forms if f.lower() != word.lower()]

    if not forms:
        return word  # fallback

    return random.choice(forms)


# ------------------------
# 3. Detect verb bằng POS 
# ------------------------
def find_verb_indices(words):
    pos_tags = nltk.pos_tag(words)
    indices = [i for i, (_, tag) in enumerate(pos_tags) if tag.startswith("VB")]
    return indices, pos_tags


# =========================
# CÁC LOẠI LỖI CỤ THỂ
# =========================

def insert_error(words, idx, deletion_candidates):
    """
    InsertError: Chèn một từ thừa (lấy từ danh sách deletion_candidates)
    vào trước vị trí idx. Danh sách này gồm các từ thường bị xóa,
    nên việc chèn chúng tạo ra lỗi "từ thừa" tự nhiên.
    """
    extra_word = random.choice(deletion_candidates)
    words.insert(idx, extra_word)
    return words


def delete_error(words, idx):
    """
    DeleteError: Loại bỏ từ tại vị trí idx khỏi câu.
    Chỉ xóa nếu câu có hơn 1 từ để giữ câu không rỗng.
    """
    if len(words) > 1:
        words.pop(idx)
    return words


def replace_error(words, idx, deletion_candidates):
    """
    ReplaceError: Kết hợp DeleteError + InsertError tại cùng vị trí.
    Từ tại idx bị xóa, sau đó một từ mới từ deletion_candidates được chèn vào.
    Kết quả: từ tại vị trí idx được thay thế bằng một từ khác.
    """
    if len(words) == 0:
        return words
    # Xóa từ hiện tại
    words.pop(idx)
    # Chèn từ mới vào đúng vị trí đó
    new_word = random.choice(deletion_candidates)
    words.insert(idx, new_word)
    return words


def verb_error(words):
    """
    VerbError: Tìm các động từ trong câu, chọn ngẫu nhiên một động từ,
    thay thế bằng một dạng chia khác của chính động từ đó.
    Nếu câu không có động từ, không thay đổi gì.
    """

    # POS tagging - hàm gán nhãn từ loại
    pos_tags = nltk.pos_tag(words)

    # Lấy index verb chuẩn - tìm vị trí các động từ
    verb_indices = [i for i, (_, tag) in enumerate(pos_tags) if tag.startswith("VB")]

    if not verb_indices:
        return words, None  # Không có verb

    # chọn 1 verb
    chosen_idx = random.choice(verb_indices)

    # trả về dạng (word, tag)
    original_verb, tag = pos_tags[chosen_idx]

    # inflect đúng (có lemma bên trong)
    words[chosen_idx] = inflect_verb(original_verb, tag)

    return words, chosen_idx


# =========================
# HÀM TẠO LỖI CHÍNH (A.3)
# =========================
def introduce_errors(
    sentence,
    deletion_candidates,
    error_count_probs=(0.05, 0.07, 0.25, 0.35, 0.28),
    error_type_probs=(0.30, 0.25, 0.25, 0.20),
):
    """
    Thêm lỗi ngữ nghĩa vào câu theo 4 loại:
      0 = InsertError  – chèn từ thừa từ deletion_candidates
      1 = VerbError – xóa từ tại vị trí ngẫu nhiên
      2 = ReplaceError – thay từ tại vị trí bằng từ trong deletion_candidates
      3 = DeleteError     – thay động từ bằng dạng chia khác

    Trả về: (noisy_sentence, labels) - labels là list 0/1 cho từng token
    """
    words = sentence.split()
    labels = [0] * len(words)  # Khởi tạo labels cho tất cả tokens

    # Số lỗi sẽ được thêm vào câu (lấy mẫu từ phân phối)
    error_count = multinoulli(list(error_count_probs))

    for _ in range(error_count):
        if len(words) == 0:
            break

        error_type = multinoulli(list(error_type_probs))

        if error_type == 0:
            # InsertError: chèn từ thừa
            idx = random.randint(0, len(words)-1)
            words = insert_error(words, idx, deletion_candidates)
            labels.insert(idx, 1)  # Chèn label 1 cho từ mới

        elif error_type == 1:
            # VerbError: chia động từ sai
            words, idx = verb_error(words)
            if idx is not None:
                labels[idx] = 1

        elif error_type == 2:
            # ReplaceError: thay thế từ
            if len(words) > 0:
                idx = random.randint(0, len(words) - 1)
                words = replace_error(words, idx, deletion_candidates)
                labels[idx] = 1  # Đánh dấu token có lỗi

        elif error_type == 3:
            # DeleteError: xóa từ
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
            labels.pop(idx)

    return " ".join(words), labels


# =========================
# XÂY DỰNG DATASET SONG SONG
# =========================
def build_parallel_dataset(clean_sentences, top_k_vocab=500):
    """
    Xây dựng bộ dữ liệu (noisy, clean, labels) từ danh sách câu sạch.
    Labels là list 0/1 cho từng token trong noisy sentence.

    Trả về: list các tuple (noisy_sentence, clean_sentence, labels)
    """
    deletion_candidates = build_deletion_candidates(
        clean_sentences, top_k=top_k_vocab
    )

    dataset = []
    for sent in tqdm(clean_sentences, desc="Generating noisy sentences"):
        noisy, labels = introduce_errors(sent, deletion_candidates)
        dataset.append((noisy, sent, labels))

    return dataset


# =========================
# DEMO / KIỂM THỬ
# =========================
if __name__ == "__main__":
    clean_data = [
        "he is playing football in the park",
        "she goes to school every morning",
        "they are eating rice and vegetables",
        "he runs very fast along the road",
        "the government has announced a new policy",
        "we should consider the environmental impact carefully",
        "How can i help you",
        "She is very beautiful and thoughful",
        "He goes to school in the morning",

    ]

    deletion_cands = build_deletion_candidates(clean_data)
    print(f"Deletion candidates (first 20): {deletion_cands[:20]}\n")

    print("=" * 60)
    print("PARALLEL DATASET (noisy → clean):")
    print("=" * 60)

    dataset = build_parallel_dataset(clean_data)
    for noisy, clean, labels in dataset:
        print(f"  CLEAN : {clean}")
        print(f"  NOISY : {noisy}")
        print(f"  LABELS: {labels}")
        print()

    # Kiểm tra từng loại lỗi riêng biệt
    print("=" * 60)
    print("INDIVIDUAL ERROR TYPE TESTS:")
    print("=" * 60)
    sample = "It gived me great pleasure to welcome Emma Bonino the italian minister for european policies and international trade to the house day."
    words = sample.split()

    print(f"Original  : {sample}")
    print(f"InsertErr : {' '.join(insert_error(list(words), 3, deletion_cands))}")
    print(f"DeleteErr : {' '.join(delete_error(list(words), 2))}")
    print(f"ReplaceErr: {' '.join(replace_error(list(words), 2, deletion_cands))}")
    print(f"VerbErr   : {' '.join(verb_error(list(words))[0])}")

