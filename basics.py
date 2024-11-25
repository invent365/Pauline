##We would specify the algorithms to use in this model.
#1)Frozen torch embeddings + ROPE positional embeddings (2) Grouped Query Attention. #(3)Confluence:optional #(4)Positional FFN
import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken,rope
base_model_encoding = tiktoken.get_encoding(tiktoken.list_encoding_names()[0]) #using gpt2

dim_head = 60 #dimension_head represents the number of attention heads the model has
embedding1 = nn.Embedding(base_model_encoding.n_vocab, dim_head) #embedding is no. of tokens in vocab * dim_head such that output = no. of sequences * no. of tokens in vocab * no. of tokens in vocab * dim_head
# = no. of sequences in dim_head.

#The idea with Pseudo_GQA is to share keys and values between all attention heads while maintaining a distinct representation for all queries. It may not be as effective as the classical model but it is deemed
#resource efficient
class Pseudo_GQA(nn.Module):
    def __init__(self, num_head, num_queries_per_group, embed_dim, dk, dv)->None:
        super(Pseudo_GQA, self).__init__()
        self.num_head = num_head #no. of heads
        self.embed_dim = embed_dim #embedding dimensions
        self.dk=dk #dimension of keys
        self.dv = dv #dimension of values
        self.num_queries_per_group = num_queries_per_group #query dimension for each group
        self.k_to_q_ratio = self.num_head/self.num_queries_per_group 
        #for every num_group, we match num_queries_per_group  queries to num_head/num_queries_per_group
        self.queries = nn.ModuleList([nn.Linear(self.embed_dim, dk) for _ in range(num_head)]) #queries Q for self attension 
        self.keys = nn.ModuleList([nn.Linear(self.embed_dim, dk*num_queries_per_group) for _ in range(int(self.k_to_q_ratio))]) 
        self.values = nn.ModuleList([nn.Linear(self.dv, self.embed_dim) for _ in range(int(self.k_to_q_ratio))])
        self.Wo = nn.Linear(self.dv*int(self.k_to_q_ratio),embed_dim) #Massive seive to recieve the output of self attention
    def forward(self, x:torch.Tensor)->list[torch.Tensor]:
        if self.num_head%self.num_queries_per_group != 0:
            raise AssertionError(f"Num_head({self.num_head}) is not divisible by num_queries_per_group({self.num_queries_per_group})")
        """Rope is a positional embedding that calculates the relative distance between two tokens and is thought to be more sensitive to relative differences in tokens compared to absolute embedding.
        we compute self_attention with queries, keys and values and pass through Wo to get our final activations. Because of the complexities involved in this model, it is primarily experimental.
        """
        out_q = [rope.yield_template(self.queries[i](x)) for i in range(len(self.queries))]
        out_q= [torch.cat(out_q[i:i+self.num_queries_per_group],dim=-1) for i in range(0, len(out_q),self.num_queries_per_group)]
        out_k = [rope.yield_template(self.keys[i](x)) for i in range(len(self.keys))]
        out_v = [rope.yield_template(self.values[i](x)) for i in range(len(self.values))]
        #mask = my_tools.mask(x.size(0), x.size(1))
        out_a = [F.scaled_dot_product_attention(query=out_q_,key=out_k_,value=out_v_,is_causal=True) for out_q_, out_k_,out_v_ in zip(out_q,out_k,out_v)]
        out_a = torch.cat(out_a, dim=-1)
        final_out_z = self.Wo(out_a)
        return final_out_z


class RMSLayerNorm(nn.Module):
    def __init__(self,expected_shape:list[int]|None,eps=1e-5):
        """
            Typical RMS Layer Norm
        """
        super(RMSLayerNorm, self).__init__()
        self.layer_norm = nn.Linear(expected_shape,expected_shape) if expected_shape else None
        self.eps = eps  
    def forward(self, x:torch.Tensor):
        k= (x/((torch.sum(x**2)/x.size(0)).sqrt())) + self.eps
        if self.layer_norm != None:
            return self.layer_norm(k)
        else:
            return k
class Position_Feedforward(nn.Module):
    def __init__(self, embed_dim:int, num_parameters:int):
        super(Position_Feedforward, self).__init__()
        #simple feedfroward that uses GELU activation. RELU could die hence our preference for GELU
        self.lin1= nn.Linear(embed_dim, num_parameters)
        self.lin2 = nn.Linear(num_parameters, embed_dim)
    def forward(self,x:torch.Tensor):
        out = F.gelu(self.lin1(x))
        out = self.lin2(out)
        return out

class Residual_Block(nn.Module): #No longer implemented. Commented code can be found in transformer_block submodule in Transformer_block Module
    def __init__(self, embed_dim:int, length:int):
        super(Residual_Block, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _ in range(length)])
    def forward(self, input:torch.Tensor):
        cache = input
        for net in self.linear:
            cache = net(cache)
        cache = cache + input
        return cache

