import os
import torch
import torch.nn as nn


def createLatentStateEncoder(dim=512):
    return LatentStateEncoder(d_model=dim)


class LatentStateEncoder(nn.Module):
    """
    Re-encodes the previous iteration's latent state matrix so it can be
    fed back into the encoding pipeline as a conditioning signal.

    Accepts the state produced at the end of one forward pass — shape
    [1, K, D] — and contextualises it with self-attention so the router
    and experts can condition on what was already computed.  The output
    is [1, K, D], fully compatible with EncoderRegistry / EncodingCombination.

    Dimension rules (same as the rest of the model):
        d_model         : D  — operates on the full latent dimension directly,
                               no pre-stage D/2 since this is already a latent
        dim_feedforward : 4*D
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, state):
        """
        Args:
            state : [1, K, D] — latent state from the previous iteration

        Returns:
            [1, K, D] — contextualised state tokens ready for EncodingCombination
        """
        return self.transformer(state)

    def store_weights(self, path, filename="latent_state_encoder.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
