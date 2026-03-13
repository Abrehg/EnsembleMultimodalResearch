import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

def createTextOutputExpert(max_seq_len=10000, vocab_size=32000, d_model=512):
    return TextOutputExpert(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len
    )

class TextOutputExpert(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=PAD_ID,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.output_proj.weight = self.token_embedding.weight

    def _sincos_pos_embed(self, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(
            -torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (math.log(10000.0) / self.d_model)
        )
        out = pos[:, None] * omega[None, :]
        emb = torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)
        return emb.unsqueeze(0)

    @staticmethod
    def _causal_mask(seq_len, device):
        return nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

    def forward(self, state, max_length=None, temperature=1.0, top_k=50):
        """
        Autoregressively decode the latent state into token IDs.
 
        Args:
            state       : [B, K, d_model]
            max_length  : cap on generated tokens (defaults to max_seq_len)
            temperature : sampling temperature (1.0 = unchanged, <1 = sharper)
            top_k       : keep only top-k logits before sampling (0 = off)
 
        Returns:
            token_ids : [B, T']  — generated token id sequences, pass to
                                   tokenizer.detokenize() to get strings
        """
        if max_length is None:
            max_length = self.max_seq_len
 
        B = state.size(0)
        device = state.device
 
        generated = torch.full(
            (B, 1), BOS_ID, dtype=torch.long, device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
 
        for _ in range(max_length):
            T = generated.size(1)
            tgt = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt = tgt + self._sincos_pos_embed(T, device)
            tgt_mask = self._causal_mask(T, device)
 
            decoded = self.decoder(
                tgt=tgt,
                memory=state,
                tgt_mask=tgt_mask,
            )
 
            logits = self.output_proj(decoded[:, -1, :])
 
            logits = logits / max(temperature, 1e-8)
 
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")
 
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
 
            next_token[finished] = PAD_ID
            generated = torch.cat([generated, next_token], dim=1)
 
            finished = finished | (next_token.squeeze(-1) == EOS_ID)
            if finished.all():
                break
 
        return generated

    def store_weights(self, path, filename="text_output_expert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)