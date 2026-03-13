import os
import torch
import torch.nn as nn

def createEncRegistry():
    return EncoderRegistry()

class EncoderRegistry(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model

        self.encoders = nn.ModuleDict()
        self.modality_embeddings = nn.ParameterDict()

    def add_encoder(self, modality: str, module: nn.Module):
        if modality in self.encoders:
            raise ValueError(
                f"Modality '{modality}' already registered. "
                f"Remove it first with remove_encoder('{modality}')."
            )

        self.encoders[modality] = module
        self.modality_embeddings[modality] = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )

        print(f"Registered encoder for modality '{modality}'")

    def remove_encoder(self, modality: str):
        """Remove an encoder and its modality embedding."""
        if modality not in self.encoders:
            raise ValueError(f"Modality '{modality}' is not registered.")

        del self.encoders[modality]
        del self.modality_embeddings[modality]

        print(f"Removed encoder for modality '{modality}'. "
              f"{len(self.encoders)} encoder(s) remaining.")

    @property
    def modalities(self):
        return list(self.encoders.keys())

    def get_encoder(self, modality: str):
        if modality not in self.encoders:
            raise ValueError(
                f"No encoder registered for modality '{modality}'. "
                f"Available: {self.modalities}"
            )
        return self.encoders[modality]

    def encode(self, raw_input, modality: str, encodings: list = None):
        if encodings is None:
            encodings = []

        encoder = self.get_encoder(modality)
        mod_emb = self.modality_embeddings[modality]

        if isinstance(raw_input, torch.Tensor):
            items = [raw_input[i].unsqueeze(0) for i in range(raw_input.size(0))]
        elif isinstance(raw_input, list):
            items = raw_input
        else:
            items = [raw_input]
 
        for item in items:
            output = encoder(item)
            output = output + mod_emb
            encodings.append(output)

        return encodings

    def store_weights(self, path, filename="encoder_registry"):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "modalities": self.modalities,
        }
        torch.save(checkpoint, os.path.join(path, filename))
        print(f"Saved encoder registry ({len(self.encoders)} encoders) "
              f"to {path}/{filename}")

    def load_weights(self, filepath, strict=True):
        checkpoint = torch.load(filepath, map_location="cpu")

        if strict:
            saved = checkpoint["modalities"]
            current = self.modalities
            if saved != current:
                raise ValueError(
                    f"Modality mismatch.\n"
                    f"  Checkpoint : {saved}\n"
                    f"  Current    : {current}\n"
                    f"Use strict=False to load partial weights."
                )

        missing, unexpected = self.load_state_dict(
            checkpoint["state_dict"], strict=strict
        )

        status = "strict" if strict else "partial"
        print(f"Loaded encoder registry weights ({status}) from {filepath}")

        if not strict:
            if missing:
                print(f"  New (randomly initialised): {len(missing)} params")
            if unexpected:
                print(f"  Skipped (removed):          {len(unexpected)} params")