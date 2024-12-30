# Section 3.4 - Implementing self-attention with trainable weights.
# NOTE: Self attention is also called 'scaled dot-product attention'.

"""
The trainable matrices are implemented using the random projection matrices - W_q, W_k, W_v.
"""

import torch
torch.manual_seed(123)
vocab_size = 6
output_dim = 3
inputs = torch.rand((vocab_size, output_dim))

x_2 = inputs[2]
d_in = inputs.shape[1]
d_out = 2
# NOTE: in GPT-like models, the input and the output dimensions are the same.
# We use smaller dimensios for simplicity.
 
# Initialize the trainable projection matrices but set requires_grad=False for now to reduce clutter.
W_query = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)

# Let's calculate the query, key, and value vectors for x_2 - just to understand how it works.
query_2 = x_2 @ W_query # Shape: (d_out,). The shapes must match.
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("Query for x_2: ", query_2)

# Now let's calculate the key and value vectors for all the input vectors with respect to x_2.
keys = inputs @ W_key
values = inputs @ W_value


# Now let's calculate the attention scores, between all the x_2 query vector and the second key vector.
keys_2 = keys[2]
attn_score_22 = torch.dot(query_2, keys_2)
print("Attention score between x_2 and the second key vector: ", attn_score_22)


# Now let's calculate the attention scores between x_2 and all the key vectors.
attn_scores_2 = query_2 @ keys.T # Shape: 1x2 X (6x3 X 3x2) = 1x2 X 2x6 = 1x6.
print("Attention scores between x_2 and all the key vectors: ", attn_scores_2)
print("Shape of attention scores: ", attn_scores_2.shape)


# Now normalize the attention scores to get the attention weights.
d_k = keys.shape[-1] # This is the dimension of the key vectors.
attn_weights_2 = torch.softmax(attn_scores_2 / (d_k ** 0.5), dim=-1) # taking the square root is the same as raising to the power of 0.5.
print("Normalized attention scores: ", attn_weights_2)

# The final step is to multiply the attention weights with the value vectors to get the context vector.
context_vector_2 = attn_weights_2 @ values # This is a multiply and sum operation.
print("Context vector for x_2: ", context_vector_2)
