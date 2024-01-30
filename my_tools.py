#masked tests
import torch, torch.nn as nn,torch.nn.functional as F

def mask(batch:int, seq_length:int):
    g=  torch.ones((batch, seq_length,seq_length))
    mask = (g.triu(0)==0)
    return mask


