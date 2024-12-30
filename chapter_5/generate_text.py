import sys
import json
import torch.nn as nn
import torch
sys.path.append("D:\llms_from_scratch\\")
sys.path.append("D:\llms_from_scratch\\chapter_4\\")
from chapter_4.gpt_model import GPTModel, generate_text_simple
import tiktoken

with open('D:\llms_from_scratch\chapter_4\gpt_config.json', 'r') as f:
    cfg = json.load(f)
    
cfg = cfg['GPT_CONFIG_124M']

torch.manual_seed(123)
model = GPTModel(cfg)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add the batch dimension.
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding('gpt2')
print("tokenized text: ", text_to_token_ids(start_context, tokenizer))

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=cfg["context_length"]
)

print("Output token ids: ", token_ids)
print("Output text: ", token_ids_to_text(token_ids, tokenizer))

# Pre-mapped inputs.
inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345],
                        [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
# print("Probas: ", probas)
# print("Probas shape: ", probas.shape)
# token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs: ", token_ids)

# # Convert the token ids back to text.
# text = token_ids_to_text(token_ids[0].flatten(), tokenizer)
# print("Text: ", text)


# Example of manual loss calculation.
text_idx = 0 # first sample.
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Target probabilities 1 : ", target_probas_1)

# NOTE: The third set of indices retireve the probabilitie values for the corresponding token IDs.

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Target probabilities 2: ", target_probas_2)

log_probas = torch.log(torch.stack((target_probas_1, target_probas_2)))
print("Result: ", log_probas)

avg_log_probas = torch.mean(log_probas)
print("Average log probabilities: ", avg_log_probas)

# Next we get the negative the mean log probabilities.
neg_avg_log_probas = avg_log_probas * -1
print("Cross entropy loss: ", neg_avg_log_probas)


# Calculating the same loss using PyTroch cross entropy loss.
print("Logits shape: ", logits.shape)
print("Targets shape: ", targets.shape)

# Now we flattent the logits and the targets.
logits_flat = logits.flatten(0,1) # Flatten along the first and second dimensions.
targets_flat = targets.flatten()

loss = nn.functional.cross_entropy(logits_flat, targets_flat)
print("Cross entropy loss: ", loss)

# Generally perplexity is used instead of raw loss
# because perplexity is a more interpretable.