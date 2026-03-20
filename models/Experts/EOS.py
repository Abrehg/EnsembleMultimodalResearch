import torch
from torch import nn

def createEOSExpert():
    return EOSExpert()

class EOSExpert(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
