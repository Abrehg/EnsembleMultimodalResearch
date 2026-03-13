import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.gelu(x + residual)

def createVisionEncoder():
    return VisionEncoder()

class VisionEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, patch_size=16, internal_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.deep_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResBlock(256),
            
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            ResBlock(embed_dim),
            ResBlock(embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu",
            batch_first=True
        )
        self.spatial_contextualizer = nn.TransformerEncoder(encoder_layer, num_layers=internal_layers)

    def _pad_to_multiple(self, x):
        H, W = x.shape[-2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    def _get_2d_sincos_pos_embed(self, grid_h, grid_w, device):
        half_dim = self.embed_dim // 2
        pos_h = torch.arange(grid_h, dtype=torch.float32, device=device)
        pos_w = torch.arange(grid_w, dtype=torch.float32, device=device)
        omega = torch.exp(-torch.arange(0, half_dim, 2, dtype=torch.float32, device=device) * (torch.log(torch.tensor(10000.0)) / half_dim))
        
        out_h = pos_h[:, None] * omega[None, :]
        out_w = pos_w[:, None] * omega[None, :]
        
        emb_h = torch.stack([torch.sin(out_h), torch.cos(out_h)], dim=2).flatten(1)
        emb_w = torch.stack([torch.sin(out_w), torch.cos(out_w)], dim=2).flatten(1)
        
        return torch.cat([
            emb_h.unsqueeze(1).expand(grid_h, grid_w, -1),
            emb_w.unsqueeze(0).expand(grid_h, grid_w, -1)
        ], dim=-1)

    def _get_1d_sincos_pos_embed(self, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(-torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) * (torch.log(torch.tensor(10000.0)) / self.embed_dim))
        out = pos[:, None] * omega[None, :]
        return torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)

    def forward(self, x):
        is_video = x.dim() == 5
        if not is_video:
            x = x.unsqueeze(1) 
            
        B, T, C, H, W = x.shape
        device = x.device
        
        x = self._pad_to_multiple(x) 
        x = x.view(B * T, C, x.shape[-2], x.shape[-1])
        
        features = self.deep_stem(x)
        _, _, grid_h, grid_w = features.shape
        
        features = features.flatten(2).transpose(1, 2)
        
        spatial_pos_emb = self._get_2d_sincos_pos_embed(grid_h, grid_w, device)
        spatial_pos_emb = spatial_pos_emb.view(1, grid_h * grid_w, self.embed_dim)
        features = features + spatial_pos_emb
        
        features = self.spatial_contextualizer(features)
        
        features = features.view(B, T, grid_h * grid_w, self.embed_dim)
        temporal_pos_emb = self._get_1d_sincos_pos_embed(T, device)
        temporal_pos_emb = temporal_pos_emb.view(1, T, 1, self.embed_dim)
        features = features + temporal_pos_emb
        
        output = features.reshape(B, -1, self.embed_dim)
        
        return output
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
# # Initialize encoder
# encoder = createVisionEncoder()

# # 1. Processing an Image (Not strictly a multiple of 16)
# # Height=250, Width=300 -> Will pad to 256x304
# dummy_image = torch.randn(2, 3, 250, 300) 
# img_embeddings = encoder(dummy_image)
# print(f"Image output shape: {img_embeddings.shape}") 
# # seq_len = (256/16) * (304/16) = 16 * 19 = 304
# # Expected output: [2, 304, 512]

# # 2. Processing a Video (Not strictly a multiple of 16)
# # 10 frames, Height=100, Width=100 -> Will pad to 112x112
# dummy_video = torch.randn(4, 10, 3, 100, 100) 
# vid_embeddings = encoder(dummy_video)
# print(f"Video output shape: {vid_embeddings.shape}") 
# # seq_len = 10 (frames) * (112/16) * (112/16) = 10 * 7 * 7 = 490
# # Expected output: [4, 490, 512]