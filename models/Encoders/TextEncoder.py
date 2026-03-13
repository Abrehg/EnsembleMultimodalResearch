import sentencepiece as spm
import torch
import torch.nn as nn
import math
from datasets import load_dataset
import sentencepiece as spm
import os

def build_spm_corpus(output_file="train_corpus.txt", sample_size=500000):
    print("Downloading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    print(f"Writing {sample_size} lines to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        count = 0
        for item in dataset:
            text = item["text"].strip()
            if len(text) > 10: 
                f.write(text + "\n")
                count += 1
            if count >= sample_size:
                break
                
    print(f"Corpus built! Saved to {output_file}")

def train_production_spm(corpus_file="train_corpus.txt", vocab_size=32000):
    print(f"Training SentencePiece model (Vocab Size: {vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=corpus_file, 
        model_prefix='multimodal_bpe', 
        vocab_size=vocab_size, 
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>'
    )
    print("Training complete! 'multimodal_bpe.model' is ready.")

# build_spm_corpus()
# train_production_spm()

def createTokenizer():
    return InputTextTokenizer()

def createTextEnc():
    return TextEncoder(vocab_size=32000, embed_dim=512)

class InputTextTokenizer:
    _DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path="multimodal_bpe.model"):
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            model_path = os.path.join(self._DIR, model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def tokenize_single(self, text, device='cpu'):
        encoded = [self.sp.bos_id()] + self.sp.encode_as_ids(text) + [self.sp.eos_id()]
        
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        padding_mask = None 
        
        return input_ids, padding_mask
    
    def detokenize(self, token_ids, strip_special=True):
        SPECIAL_IDS = {self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()}
 
        # Normalise input to a list of lists
        single = False
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                single = True
                token_ids = token_ids.unsqueeze(0)
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list) and token_ids and not isinstance(token_ids[0], list):
            single = True
            token_ids = [token_ids]
 
        results = []
        for ids in token_ids:
            if strip_special:
                ids = [i for i in ids if i not in SPECIAL_IDS]
            results.append(self.sp.decode_ids(ids))
 
        return results[0] if single else results
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim, 
            padding_idx=0
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_1d_sincos_pos_embed(self, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(-torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) * (math.log(10000.0) / self.embed_dim))
        
        out = pos[:, None] * omega[None, :]
        emb = torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)
        
        return emb.unsqueeze(0)

    def forward(self, input_ids, padding_mask=None):
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.token_embedding(input_ids) * math.sqrt(self.embed_dim)
        
        pos_emb = self._get_1d_sincos_pos_embed(seq_len, device)
        x = x + pos_emb
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        return x
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
    
# # Assuming you have loaded the InputTextTokenizer and TextEncoder
# tokenizer = createTokenizer()
# encoder = createTextEnc()

# # Test 1: A short tweet
# tweet = "Just set up my new multimodal PyTorch model! 🚀"
# tweet_ids, _ = tokenizer.tokenize_single(tweet)
# tweet_embeds = encoder(tweet_ids)

# print(f"Tweet input shape: {tweet_ids.shape}")
# print(f"Tweet output shape: {tweet_embeds.shape}") 
# # Expected: [1, ~15, 512]

# # Test 2: A long article
# article = "PyTorch is an open source machine learning library... " * 1000
# article_ids, _ = tokenizer.tokenize_single(article)
# article_embeds = encoder(article_ids)

# print(f"Article input shape: {article_ids.shape}")
# print(f"Article output shape: {article_embeds.shape}") 
# # Expected: [1, ~12000, 512]