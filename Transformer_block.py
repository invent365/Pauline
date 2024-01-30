##transformer submodule
#This module  instantiates the Transformer Block up until the level of an instantiable transformer
import torch, torch.nn as nn,torch.nn.functional as F
import basics
import tiktoken

base_model_encoding = tiktoken.get_encoding(tiktoken.list_encoding_names()[0]) #using gpt2

embed_dim, num_heads,num_params, num_queries_per_group,num_blocks = 512, 12, 2048,4,10
embedding1 = nn.Embedding(base_model_encoding.n_vocab, embed_dim)

class Transformer_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, num_parameters:int, num_queries_per_group:int,dk=embed_dim,dv=embed_dim):
        super(Transformer_Block,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk,self.dv = dk,dv
        self.transformer_multihead = basics.Pseudo_GQA(num_head=num_heads, num_queries_per_group=num_queries_per_group,embed_dim=self.embed_dim,dk=self.dk,dv=self.dv)
        self.rms = basics.RMSLayerNorm(None)
        self.ffn = basics.Position_Feedforward(self.embed_dim,num_parameters)
        #self.res_block = basics.Residual_Block(self.embed_dim, length=length_of_res_block)
    def forward(self,x:torch.Tensor):
        normed_input = self.rms(x)
        out = self.transformer_multihead(normed_input)
        out = x + out # add_norm_1
        normed_output2 = self.rms(out)
        normed_output2 = self.ffn(normed_output2)
        out = out + normed_output2
        return out
    
class Transformer(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, num_parameters:int,num_queries_per_group:int, num_blocks:int,dk=embed_dim,dv=embed_dim):
        super(Transformer,self).__init__()
        self.T_stacked = nn.ModuleList([Transformer_Block(embed_dim,num_heads,num_parameters,num_queries_per_group,dk,dv,) for _ in range(num_blocks)])
    def print_params(self):
        print(sum([p.numel() for p in self.parameters()]))
    def forward(self, x:torch.Tensor):
        for i in range(len(self.T_stacked)):
            x = self.T_stacked[i](x)
        return x






