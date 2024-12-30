import torch.nn as nn 
import torch
torch.manual_seed(123)
class SelfAttentionV1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttentionV1, self).__init__()
        self.W_query = nn.Parameter(torch.rand((in_dim, out_dim)), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand((in_dim, out_dim)), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand((in_dim, out_dim)), requires_grad=True)
        
    
    def forward(self, x):
        
        # Works for one or a batch of input vectors.
        
        # Check if the input.shape[-1] is equal to the in_dim.
        assert x.shape[-1] == self.W_query.shape[0], "Input dimension does not match the query dimension."
        queries = x @ self.W_query # Shape: x.shape[0] x out_dim.
        keys = x @ self.W_key # Shape: x.shape[0] x out_dim.
        values = x @ self.W_value # Shape: x.shape[0] x out_dim.
        
        # Calculate the attention scores.
        scores = queries @ keys.T # Shape: (x.shape[0] x out_dim) x (out_dim x x.shape[0]) = x.shape[0] x x.shape[0].
        normalized_scores = torch.softmax(scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vectors = normalized_scores @ values # Shape: x.shape[0] x out_dim.
        
        return context_vectors
    
# Let's implement self-attention using nn.Linear.
# Using nn.Linear has the benefit of optimized weight initialization and bias initialization. 


class SelfAttentionV2(nn.Module):
    def __init__(self, in_dim, out_dim, qkv_bias=False):
        super(SelfAttentionV2, self).__init__()
        self.W_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        
        
    def forward(self, x):
        
        # Check shapes.
        assert x.shape[-1] == self.W_query.in_features, "Input dimension does not match the query dimension."
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        scores = queries @ keys.T # Attention scores. 
        normalized_scores = torch.softmax(
            scores / (keys.shape[-1] ** 0.5), dim=-1
        )    # Attention weights.
        
        context_vectors = normalized_scores @ values
        
        return context_vectors

 
# Testing the class.
vocab_size = 6
output_dim = 3
inputs = torch.rand((vocab_size, output_dim))
input_dim = inputs.shape[-1]
output_dim = 2
sa_v1 = SelfAttentionV1(input_dim, output_dim)
print("Inputs: ", inputs)
print("Context vectors v1: ", sa_v1(inputs))


# Test SelfAttentionV2. This will produce different results because nn.Linear initializes the weights differently.
sa_v2 = SelfAttentionV2(input_dim, output_dim)
print("Context vectors v2: ", sa_v2(inputs))


# Exercise 3.1. Weight transfer from V2 to V1 to produce identical results.
sa_v2.W_query.weight.data = sa_v1.W_query.T
sa_v2.W_key.weight.data = sa_v1.W_key.T
sa_v2.W_value.weight.data = sa_v1.W_value.T

print("Context vectors v2 after weight transfer: ", sa_v2(inputs))

"""
Let's implement Causal Self-Attention <- For tasks such as language modeling.
"""
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attention_scores = queries @ keys.T # Shape: (vocab_size x output_dim) x (output_dim x vocab_size) = vocab_size x vocab_size.
attn_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)
print("Attention weights: ", attn_weights)

# Now let's mask out the future tokens.
context_length = attention_scores.shape[0] # This is the length of the context. NOTE: This is not the vocab.
mask_simple = torch.tril(torch.ones(context_length, context_length)) # Lower triangular matrix.

# Now apply the lower triangular mask to the 'attn_weights' (not the attn_scores).
# NOTE: This is a element wise multiplication.
masked_attention_weights = attn_weights * mask_simple # This will zero out the future tokens.
print("Masked attention scores: ", masked_attention_weights)

# Let's re-normalize the attention weights, we don't use softmax here.
row_sums = masked_attention_weights.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_attention_weights / row_sums
print("Masked simple normalized attention scores: ", masked_simple_norm)

# Now let's impleemnt a more efficient way to mask out the future tokens using -inf.
# Using torch.triu instead of torch.tril.
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) # Lower triangular matrix.
# NOTE: the 'attention_scores' are masked and not the 'attn_weights'.
masked_with_inf = attention_scores.masked_fill(mask.bool(), -torch.inf) # Replace the masked values with -inf where value is 1.
print("Masked attention scores with -inf: ", masked_with_inf)
attn_weights_with_inf = torch.softmax(masked_with_inf / (keys.shape[-1] ** 0.5), dim=1)
print("Attention weights with -inf: ", attn_weights_with_inf)

# The above is the causal self-attention mechanism.
# We can directly modify the class with the efficient masking technique
# but let's implement a mechanism for reducing overfitting when training LMs.

# Using dropout.
dropout = torch.nn.Dropout(0.1)
# Apply dropout to the attention weights (not the scores).
attn_weights_with_dropout = dropout(attn_weights_with_inf)
print("Attention weights with dropout: ", attn_weights_with_dropout)