# Now we need to create embeddings for the token IDs in the input and target tensors.
# Initially the embeddings are randomly initialized. Then they are updated during training.

import torch
import numpy as np

input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
vocab_size = 14 # Maximum token ID + 1
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("Embedding layer: ", embedding_layer.weight)


# Now let's apply the embeddings to the input tensor.
input_embeddings = embedding_layer(input_ids)
print("Input embeddings: ", input_embeddings)