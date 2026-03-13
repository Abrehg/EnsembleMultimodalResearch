import torch
import torch.nn as nn
import os

class AdaptiveAttentionPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_queries: int = 128,
        min_queries: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        gate_steepness: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_queries = max_queries
        self.min_queries = min_queries
        self.gate_steepness = gate_steepness

        self.query_tokens = nn.Parameter(
            torch.randn(1, max_queries, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _mean_pool(self, x, mask=None):
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return x.mean(dim=1)

    def forward(self, encoder_out, key_padding_mask=None, total_seq_len=None):
        B, S, D = encoder_out.shape

        effective_max = self.max_queries
        if total_seq_len is not None:
            effective_max = min(self.max_queries, total_seq_len)
        effective_min = min(self.min_queries, effective_max)

        summary = self._mean_pool(encoder_out, key_padding_mask)
        complexity = torch.sigmoid(self.complexity_head(summary))

        num_active_float = (
            effective_min
            + complexity * (effective_max - effective_min)
        )

        positions = torch.arange(
            effective_max, device=encoder_out.device, dtype=encoder_out.dtype
        )

        gates = torch.sigmoid(
            self.gate_steepness * (num_active_float - positions - 0.5)
        )

        queries = self.query_tokens[:, :effective_max, :].expand(B, -1, -1)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=key_padding_mask,
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))

        queries = queries * gates.unsqueeze(-1)
        
        num_active_int = num_active_float.round().long().squeeze(-1)
        num_active_int = num_active_int.clamp(
            min=effective_min, max=effective_max
        )

        K_trim = num_active_int.max().item()
        latents = queries[:, :K_trim, :]

        return latents, num_active_int

def createEncCombine():
    minTokens = 4
    maxTokens = 128
    return EncodingCombination(min_latent_tokens=minTokens, max_latent_tokens=maxTokens)

class EncodingCombination(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_latent_tokens: int = 128,
        min_latent_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.pool = AdaptiveAttentionPooling(
            d_model=d_model,
            max_queries=max_latent_tokens,
            min_queries=min_latent_tokens,
            nhead=nhead,
            dropout=dropout,
        )

    def forward(self, input_tensors, padding_masks=None):
        total_seq_len = sum(t.size(1) for t in input_tensors)
        combined_seq = torch.cat(input_tensors, dim=1)
        combined_mask = (
            torch.cat(padding_masks, dim=1)
            if padding_masks is not None
            else None
        )
        memory = self.encoder(
            combined_seq, src_key_padding_mask=combined_mask
        )

        return self.pool(
            memory,
            key_padding_mask=combined_mask,
            total_seq_len=total_seq_len,
        )
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

# from TextEncoder import createTextEnc, createTokenizer
# from VisionEncoder import createVisionEncoder

# # Initialize encoders
# imgEncoder = createVisionEncoder()
# tokenizer = createTokenizer()
# textEncoder = createTextEnc()

# dummy_image = torch.randn(2, 3, 250, 300) 
# img_embeddings = imgEncoder(dummy_image)
# print(f"Image output shape: {img_embeddings.shape}")

# dummy_video = torch.randn(4, 10, 3, 100, 100) 
# vid_embeddings = imgEncoder(dummy_video)
# print(f"Video output shape: {vid_embeddings.shape}")

# tweet = "Just set up my new multimodal PyTorch model! 🚀"
# tweet_ids, _ = tokenizer.tokenize_single(tweet)
# tweet_embeds = textEncoder(tweet_ids)
# print(f"Tweet output shape: {tweet_embeds.shape}")

# article = "PyTorch is an open source machine learning library... " * 100 
# article_ids, _ = tokenizer.tokenize_single(article)
# article_embeds = textEncoder(article_ids)
# print(f"Article output shape: {article_embeds.shape}")

# combine = createEncCombine()

# inputs = [img_embeddings[0], img_embeddings[1], vid_embeddings[0], vid_embeddings[1], vid_embeddings[2], vid_embeddings[3], tweet_embeds[0], article_embeds[0]]
# print(f"Input shape: {len(inputs)}")

# combinedEnc, num_active = combine(inputs, modalities)
# print(f"Output shape: {combinedEnc.size()}")
# # Should be [1, ~64, 512]