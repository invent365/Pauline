#masked tests
import torch, torch.nn as nn,torch.nn.functional as F

g =torch.rand((8,8))
print(g)
mask = (g.tril(0)==0)
print(mask)
g = g.masked_fill(mask, float("-inf"))
print(g)
