import torch
import torch.nn as nn
torch.manual_seed(123)

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, 
                 context_length, dropout_value=0.1, num_heads=2, qkv_bias=False):
        super(EfficientMultiHeadAttention, self).__init__()
        
        # Check if the output dimension is divisible by the number of heads.
        # We do this to see if the output dimension can be split evenly across heads.
        assert out_dim % num_heads == 0, "Output dimension must be divisible by the number of heads."
        
        self.d_out = out_dim
        self.num_heads = num_heads
        self.head_out_dim = out_dim // num_heads
        
        self.W_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(out_dim, out_dim) # This is a linear layer to combine head outputs.
        
        self.dropout = nn.Dropout(dropout_value)
       
        # Mask diagonal should be 1. Lower triangular part is zero. 
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        
        
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        
        # Calculate key, query, and value vectors.
        # The shape is batch_size x num_tokens x d_out.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Split the key, query, and value vectors into multiple heads.
        # The last two values are correct because self.head_out_dim= out_dim // num_heads.
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_out_dim)
        
        # Now we have to calculate the attention scores for each head.
        # But wait! We need to transpose the dimensions to make the matrix multiplication work.
        
        queries = queries.transpose(1, 2) # batch_size x num_heads x num_tokens x head_out_dim.
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Computes the attention score for each head.
        attention_scores = queries @ keys.transpose(2, 3) # We need to transpose to make multiplication work.
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # This is okay because we are doing a calculation for each head.
        
        attention_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        
        # print("Shape of attn_weights: ", attn_weights.shape)
        
        # Apply dropout. Removing this will give you identical results across batches.
        attn_weights = self.dropout(attn_weights)
        
        
        # Get the context vectors.
        # Attention weights is of shape: batch_size x num_heads x num_tokens x num_tokens.
        # Values is of shape: batch_size x num_tokens x num_heads x head_out_dim.
        # Result is of shape: batch_size x num_heads x num_tokens x head_out_dim (after the transpose).
        context_vectors = (attn_weights @ values).transpose(1, 2)
        
        
        # Combine the head outputs.
        # Contiguous just lays out the data in a contiguous manner.
        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        
        # Optionally apply a linear layer to combine the heads.
        context_vectors = self.out_proj(context_vectors)
        
        return context_vectors
    
    
# Testing the class.
# inputs = torch.rand((6, 768))
# batch = torch.stack((inputs, inputs), dim=0) # Stack across rows.
# context_length = batch.shape[1]
# d_in = batch.shape[-1]
# d_out = d_in
# num_heads = 12
# mha = EfficientMultiHeadAttention(d_in, d_out, context_length, dropout_value=0.1,
#                                  num_heads=num_heads, qkv_bias=False)

# print("Output: ", mha(batch))
# print("Output shape: ", mha(batch).shape) # Should be 2x6x768.