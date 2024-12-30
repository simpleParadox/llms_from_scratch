# Implementing a compact version of the self-attention mechanism.
# Using causal masking and also dropout for regularization.


import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_value=0.1,
                 context_length=6, qkv_bias=False):
        super(CausalAttention, self).__init__()
        
        self.context_length = context_length
        
        self.W_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        
        self.dropout = nn.Dropout(dropout_value) 
        # Mask diagonal should be 1.
        # self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        
        # It may be worthwhile to use register buffers, which ensures that the variables 
        # are automatically moved to the appropriate device such as CPU or GPU.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        
        
    
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        assert self.context_length == num_tokens, "Context length does not match the number of tokens."
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # Transpose the second and third indices only keeping the batch in the first dimension.
        attention_scores = queries @ keys.transpose(1, 2)
        
        # Apply the mask on the attention scores
        attention_scores_masked = attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # Only apply to num_tokens because because that's the max_length of the input.
        
        # Do softmax.
        attention_weights_masked = torch.softmax(attention_scores_masked / (keys.shape[-1] **0.5), dim=-1) # On the last dimension. Do not do dim=1 here because it can be the num_tokens column.
        
        # Apply the dropout to the attention_weights_masked.
        attention_scores_masked_dropout = self.dropout(attention_weights_masked)
        
        # Apply the attention_weights_masked to the value 
        context_vectors = attention_scores_masked_dropout @ values
        
        return context_vectors
    


# Testing the class.
inputs = torch.rand((6, 3))
batch = torch.stack((inputs, inputs), dim=0) # Stack across rows.
print("Batch: ", batch)
input_dim = inputs.shape[-1]
output_dim = 2
context_length = batch.shape[1]
ca = CausalAttention(input_dim, output_dim, context_length=context_length, dropout_value=0.1)
context_vecs = ca(batch)
print("Context vectors: ", context_vecs)