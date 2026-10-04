import os

import torch
from torch import nn

def createEOSExpert():
    return EOSExpert()

class EOSExpert(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def store_weights(self, path, filename="eos_expert"):
        """EOSExpert has no learnable parameters (state_dict() is empty) --
        this exists only so callers can treat every expert uniformly (e.g.
        iterating ExpertRegistry.experts and calling store_weights on each)
        without a special case for this one."""
        from weight_io import atomic_torch_save

        os.makedirs(path, exist_ok=True)
        atomic_torch_save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath, map_location="cpu", strict=True):
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)
