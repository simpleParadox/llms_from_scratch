from compact_self_attention import CausalAttention
import torch
import torch.nn as nn

torch.manual_seed(123)
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads=2, qkv_bias=False):
        super(MultiHeadAttentionWrapper, self).__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, dropout, context_length, qkv_bias) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        res = []
        for head in self.heads:
            res.append(head(x))
        
        return torch.cat(res, dim=-1) # Concatenate across the column dimension.
    
    
    
# Testing the class.
inputs = torch.rand((6, 3))
batch = torch.stack((inputs, inputs), dim=0) # Stack across rows.

context_length = batch.shape[1]
d_in = batch.shape[-1]
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.1,
                                num_heads=2, qkv_bias=False)
context_vecs = mha(batch)
print("Output: ", context_vecs)
print("Context vecs shape: ", context_vecs.shape) # Should be 2x6x4.

# NOTE: The multi-head attention is calculated sequentially can be 
# sped up using matrix multiplication.