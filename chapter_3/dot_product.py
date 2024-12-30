import torch
import numpy as np

vocab_size = 6
output_dim = 3
inputs = torch.rand((vocab_size, output_dim))
print("Inputs: ", inputs)

# The self-attention is basically a dot product by itself.
attention_scores = torch.empty(inputs.shape[0])
print("Attention empty tensor: ", attention_scores)

# Let's calculate the dot product with respect to the second input tensor.
query = inputs[1]
for i in range(inputs.shape[0]):
    attention_scores[i] = torch.dot(query, inputs[i])
    
print("Attention scores: ", attention_scores)


# We can see that some of the attention scores are greter than 1.
# We have to normalize the attention scores to maintain training stability 
# and also to make sure the values add up to one.


normalized_attention_scores = attention_scores / attention_scores.sum()
print("Normalized attention scores: ", normalized_attention_scores)
print("Sum of normalized attention scores: ", normalized_attention_scores.sum()) # Should add up to 1.


# It is however possible that the attention scores can be negative.
# Using the simple normalization above can lead to negative values.
# We want positive attention weights to keep it more interpretable.

def softmax_naive(x: torch.Tensor):
    # The function expects a 1D tensor.
    return torch.exp(x) / torch.exp(x).sum(dim=0) # Sum along the first dimension.


naive_softmax_attention_scores = softmax_naive(attention_scores)
print("Naive: ", naive_softmax_attention_scores)
print("Sum of naive softmax attention scores: ", naive_softmax_attention_scores.sum()) # Should add up to 1.


# NOTE: The above function may encounter numerical instability problems,
# such as numerical overflow or underflow.

# Thus, we can use the softmax function from PyTorch.
softmax_attention_scores = torch.nn.functional.softmax(attention_scores, dim=0)
print("Softmax: ", softmax_attention_scores)
print("Sum of softmax attention scores: ", softmax_attention_scores.sum()) # Should add up to 1.


# Now let's calculate the context vector which is nothing but the weighted sum of the input vectors / embeddings.
# This is a efficient way to calculate the context vector.
weighted_vectors = softmax_attention_scores.unsqueeze(1) * inputs
print("Weighted vectors: ", weighted_vectors)
print("Sum of weighted vectors: ", weighted_vectors.sum(dim=0))

context_vector_1 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_1 += softmax_attention_scores[i] * x_i
    
print("Context vector 1: ", context_vector_1)

# context_vector_1 and context_vector_2 should be the same.


# Let's compute the attention scores for all the vectors.
attn_scores = inputs @ inputs.T # Matrix multiplication to get the self-attention scores.
print("Attention scores: ", attn_scores)

# Using torch.softmax to normalize the attention scores.
softmax_attention_scores = torch.softmax(attn_scores, dim=-1) # Normalize along the last dimension (column).
# This will normalize horizontally in the matrix (across the columns).

print("Softmax attention scores: ", softmax_attention_scores)

# Now let's calculate the context vector for each input vector.
context_vectors = softmax_attention_scores @ inputs
print("Context vectors: ", context_vectors)
# So there are three steps in total. 
"""
Use the @ operator to calculate the attention scores (matrix multiplication).
1. Calculate the attention scores by multiplying the input vectors with each other.
2. Normalize the attention scores using the softmax function.
3. Calculate the context vector by multiplying the normalized attention scores with the input vectors. 
"""

print("Second context vector: ", context_vectors[1]) # Second context vector.