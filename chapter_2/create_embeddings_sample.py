# Now we need to create embeddings for the token IDs in the input and target tensors.
# Initially the embeddings are randomly initialized. Then they are updated during training.

import torch
import numpy as np

# input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# vocab_size = 14 # Maximum token ID + 1
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # Randomly initialized embeddings.
# print("Embedding layer: ", embedding_layer.weight)


# # Now let's apply the embeddings to the input tensor.
# input_embeddings = embedding_layer(input_ids)
# print("Input embeddings: ", input_embeddings)


# The above code does not have positional information in them.
# We have two choices: relative positional embeddings or absolute positional embeddings..

# Let's create a more realistic example of a tokenizer.
vocab_size = 50257 # This is the vocab size of the raw_text.
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Let's first get the token ids from the GPTDatasetV1 class.
from iterate_using_dataloader import create_dataloader
from chapter_2_tokenizing_text import raw_text # This is the text from the-verdict.txt
max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: ", inputs)
print("Shape of token IDs: ", inputs.shape)

# Now embed the token ids into vectors.
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


# Now let's create positional embeddings. We will use absolute positional embeddings.
context_length = max_length # We do this because there a are max_length tokens in each input tensor.
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # Needs to be a torch tensor.
# The pos_embeddings are the absolution positional embeddings where each position has a different embedding randomly initialized.
print("Positional embeddings: ", pos_embeddings)
print("Shape of positional embeddings: ", pos_embeddings.shape)

# Now simply, we just add the pos_embeddings to the token_embeddings.
input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings: ", input_embeddings)
print("Shape of input embeddings: ", input_embeddings.shape)
