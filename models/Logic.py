import torch
import torch.nn as nn
import torch.nn.functional as F

def createRouter():
    return RoutingLogic()

class RoutingLogic(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        route_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.route_dim = route_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.route_query = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )
        self.route_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.route_norm = nn.LayerNorm(d_model)

        self.route_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, route_dim),
        )

        self.normalize = True

    def forward(self, state):
        B = state.size(0)

        state = self.state_encoder(state)

        query = self.route_query.expand(B, -1, -1)

        attn_out, _ = self.route_cross_attn(
            query=query,
            key=state,
            value=state,
        )
        route_token = self.route_norm(query + attn_out)
        route_token = route_token.squeeze(1)

        route_vec = self.route_proj(route_token)

        if self.normalize:
            route_vec = F.normalize(route_vec, dim=-1)

        return route_vec, state
    
# from Encoders.TextEncoder import createTextEnc, createTokenizer
# from Encoders.VisionEncoder import createVisionEncoder
# from Encoders.EncCombine import createEncCombine

# # Initialize encoders
# imgEncoder = createVisionEncoder()
# tokenizer = createTokenizer()
# textEncoder = createTextEnc()
# combine = createEncCombine()

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

# inputs = [img_embeddings[0], img_embeddings[1], vid_embeddings[0], vid_embeddings[1], vid_embeddings[2], vid_embeddings[3], tweet_embeds[0], article_embeds[0]]
# modalities = ['vision', 'vision', 'vision', 'vision', 'vision', 'vision','text', 'text']
# combinedEnc, num_active = combine(inputs, modalities)
# print(f"Combined shape: {combinedEnc.size()}")

# router = createRouter()
# routing_vec, state = router(combinedEnc)
# print(f"New state shape: {state.size()}")
# print(f"Routing vector shape: {routing_vec.size()}")