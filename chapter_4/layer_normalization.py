import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False means we are not using Bessel's correction.
        norm = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift
    
    
# Testing the LayerNorm class
# torch.manual_seed(123)
# ln = LayerNorm(5)
# batch_example = torch.randn(2, 5)
# out = ln(batch_example)
# # Now calculate the mean and variance of the layer normalized output.
# torch.set_printoptions(sci_mode=False)
# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True, unbiased=False)
# print("Mean: ", mean)
# print("Variance: ", var)