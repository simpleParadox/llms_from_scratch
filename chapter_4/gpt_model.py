import torch
import torch.nn as nn
import sys
sys.path.append("D:\llms_from_scratch\\")
sys.path.append("D:\llms_from_scratch\\chapter_4\\")
from transformer_block import TransformerBlock
from layer_normalization import LayerNorm
import json
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg["n_layers"]
        
        # First define the token and positional embeddings.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(self.n_layers)]
        )
        
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"]) # Language modelling head.
        
    def forward(self, in_idx):
        
        # in_idx is basically the tokenized input sequence.
        # Takes in the token ids from the tokenizer.
        # Implemented for one batch.
        batch_size, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx) # It actually returns batch_size x seq_len x emb_dim.
        # import pdb; pdb.set_trace()
        positional_embeddings = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeds + positional_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits # Without softmax.
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # only get the last context_size tokens.
        with torch.no_grad():
            logits = model(idx_cond)
            
        # Get the last token from the model output. This will be used to get the probability of the next token.
        logits = logits[:, -1, :] # Get the last token.
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Get the index of the token with the highest probability.
        
        # Append the new token to the input sequence.
        idx = torch.cat((idx, idx_next), dim=-1) # Concatenate the new token to the input sequence.
        
    return idx # Return the input sequence with the new tokens appended.



        
    
    
# with open('gpt_config.json', 'r') as f:
#     cfg = json.load(f)
# cfg = cfg['GPT_CONFIG_124M']
# tokenizer = tiktoken.get_encoding('gpt2')
# # Let's test the GPTModel class.
# model = GPTModel(cfg)

# # Testing the generate_text_simple function.
# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("Input sequence: ", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add the batch dimension.
# print("Input tensor shape with batch dimension: ", encoded_tensor.shape)

# # We should be using dropout only during training.
# model.eval() # disable components like dropout during inference.
# out = generate_text_simple(
#     model, encoded_tensor,
#     max_new_tokens=20,
#     context_size=cfg["context_length"]
# )
# print("Output sequence: ", out)
# print("Output sequence shape: ", out.shape)

# decoded = tokenizer.decode(out[0].tolist())
# print("Decoded output: ", decoded)




# print("Input: ", token_ids)
# output = model(token_ids)
# print("Output shape: ", output.shape) # Should be 2x4x50257.

# total_params = sum(p.numel() for p in model.parameters())
# print("Total parameters: ", total_params)
# # The total number of parameters is 163M because we
# # do not have weight tying (that was used in the original GPT model - thus reducing the number of parameters).
# # If we subtract the out_head parameters, we get 124M parameters.


# # Weight tying reduces memory footprint, but having a separate output head
# # allows for better training and model performance.

# # Calculating the number of parameters in the feedforward layer.
# fflayer_count = 0
# for layer in model.trf_blocks:
#     fflayer_count += sum(p.numel() for p in layer.ff.parameters())
    
# print("Number of parameters in the feedforward layer: ", fflayer_count) # Should be 2.4M parameters.

# # Calculating the number of parameters in the multi-head attention layer.
# mhalayer_count = 0
# for layer in model.trf_blocks:
#     mhalayer_count += sum(p.numel() for p in layer.att.parameters())
    
    
# print("Number of parameters in the multi-head attention layer: ", mhalayer_count) # Should be 2.4M parameters.

# # Total size of the model in MB.
# total_size_bytes = total_params * 4 # 4 bytes per float.
# total_size_mb = total_size_bytes / (1024 ** 2) # Converts to MB.
# print("Total size of the model in MB: ", total_size_mb)



 