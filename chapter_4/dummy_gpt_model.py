import torch
import torch.nn as nn
import json

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Randomly initialize some embeddings for token and positional components.
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim']) # I think this are the input embeddings that are in fact learned. The token ids act as indices for specific embeddings in this embedding matrix.
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim']) # Position relative to the words in the sequence.
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        
        # Create the transformer blocks.
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size']) # Output head num_columns should have the same size as the vocab_size.
        
    def  forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # Get specific token embeddings from the embeddings matrix.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # input_embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Not put the input embeddings through the transformer blocks.
        x = self.transformer_blocks(x)
        
        x = self.final_norm(x)
        
        logits = self.out_head(x) # Before the softmax operation (softmax gives us the probabilities).
        
        return logits
    
    
class DummyTransformerBlock(nn.Module): # Need to implement the projection and attention and everything.
    
    def __init__(self, cfg):
        super().__init__()
        
        
    def forward(self, x):
        return x
        
        
class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
    def forward(self, x):
        return x
    
    
# let's prepare the input data and initialize a new GPT model to illustrate the usage.

import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = "Every day is a"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print("Batch: ", batch)


# Load configuration from the json file.
with open('gpt_config.json', 'r') as f:
    cfg = json.load(f)
    
gpt_config_124 = cfg['GPT_CONFIG_124M']

torch.manual_seed(123)
model = DummyGPTModel(gpt_config_124)
logits = model(batch)
# print("Output shape: ", logits.shape)
# print("Logits: ", logits)


# Implementing layer normalization helps to improve the stability and efficiency of training a neural network.
# Let's do an example for layer normalization.
batch_example = torch.randn(2, 5)
print("Batch example: ", batch_example)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print("Output: ", out)

# Let's examing the mean and variance:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean: ", mean)
print("Variance: ", var)

# Layer normalization consists of subtracting the mean and dividing by the square root of the variance (or std).
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print("Normalized output: ", out_norm)
print("Normalized mean: ", mean)
print("Normalized variance: ", var)
