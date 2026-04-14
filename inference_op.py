# inference_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
from models.transceiver_calibration import CA_DeepSC
from utils import SNR_to_noise, greedy_decode_calibration, SeqtoText, load_checkpoint
from performance import Similarity, BleuScore
from dataset import EurDataset, collate_pair_data
from torch.utils.data import DataLoader

app = FastAPI(title="DeepSC CA API")

# ----- CONFIG -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNR = 12  # default SNR

vocab_file = './data/vocab.json'
checkpoint_path = './kaggle/working/checkpoints/ca-deepsc-AWGN'

# Load vocab
vocab = json.load(open(vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = {v: k for k, v in token_to_idx.items()}
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]

# ----- MODEL -----
args = type('', (), {})()  # dummy args
args.num_layers = 4
args.d_model = 128
args.num_heads = 8
args.dff = 512
args.MAX_LENGTH = 30
args.channel = 'AWGN'

model = CA_DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                  args.d_model, args.num_heads, args.dff, 0.1).to(device)

checkpoint = load_checkpoint(checkpoint_path, mode='best')
if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded checkpoint.")
else:
    print("No checkpoint found.")

model.eval()

StoT = SeqtoText(token_to_idx, end_idx)
similarity_calc = Similarity(batch_size=1)
bleu_calc = BleuScore(1,0,0,0)

# ----- REQUEST BODY -----
class InputSentence(BaseModel):
    sentence: str
    snr: float = SNR

# ----- API ROUTE -----
@app.post("/predict")
def predict(input_data: InputSentence):
    try:
        # Convert sentence to tokens
        input_tokens = [start_idx] + [
            token_to_idx.get(word, token_to_idx["<UNK>"])
            for word in input_data.sentence.split()
        ] + [end_idx]
        
        if len(input_tokens) > args.MAX_LENGTH + 2:
            raise HTTPException(status_code=400, detail=f"Sentence too long. Max length {args.MAX_LENGTH}")
        
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        noise_std = SNR_to_noise(input_data.snr)
        with torch.no_grad():
            output_tokens, _ = greedy_decode_calibration(model, input_tensor,
                                                         noise_std, args.MAX_LENGTH,
                                                         pad_idx, start_idx,
                                                         args.channel, device)
        
        if isinstance(output_tokens, torch.Tensor):
            output_tokens = output_tokens.squeeze(0).cpu().numpy().tolist()
        
        # Remove special tokens
        clean_tokens = []
        for t in output_tokens:
                if t == end_idx:  # token <END>
                    break          # dừng lấy token sau <END>
                if t in (start_idx):  # bỏ <START>
                    continue
                clean_tokens.append(t)
        output_text = StoT.sequence_to_text(clean_tokens)
        
        # Metrics
        bleu = bleu_calc.compute_blue_score([input_data.sentence], [output_text])[0]
        sim = similarity_calc.compute_similarity([input_data.sentence], [output_text])[0]
        
        return {
            "input": input_data.sentence,
            "output": output_text,
            "BLEU": float(bleu),
            "Similarity": float(sim)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))