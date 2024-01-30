import Transformer_block
import torch, torch.nn as nn,torch.nn.functional as F
import tiktoken

base_model_encoding = tiktoken.get_encoding(tiktoken.list_encoding_names()[0]) #using gpt2
embed_dim, num_heads,num_params, num_queries_per_group,num_blocks = 512, 12, 2048,4,6

dummy_input = torch.LongTensor([[10,50,50,20,30,10,80,70,20,60], [5,6,7,8,9,7,5,5,4,3], [2,3,45,67,89,20,11,23,19,11]])
x =dummy_input #assuming it has gone through the weight query


class Net(nn.Module):
    def __init__(self, embed_dim,num_heads,num_params,num_queries_per_group,num_blocks):
        super(Net,self).__init__()
        self.embedding= nn.Embedding(base_model_encoding.n_vocab, embed_dim)
        self.transformer = Transformer_block.Transformer(embed_dim, num_heads,num_params,num_queries_per_group,num_blocks)
        self.linear = nn.Linear(embed_dim,base_model_encoding.n_vocab)
    def print_params(self):
        print(sum([p.numel() for p in self.parameters()]))
    def forward(self, x:torch.Tensor):
        out = self.embedding(x)
        out = self.transformer(out)
        out = self.linear(out)
        return out
net = Net(embed_dim,num_heads,num_params,num_queries_per_group, num_blocks)
net.print_params()
out = net(dummy_input)
print(out.argmax(-1))

