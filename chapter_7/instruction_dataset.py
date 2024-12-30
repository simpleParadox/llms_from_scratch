import torch
import json
from torch.utils.data import Dataset, DataLoader
from format_data import format_input
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
torch.manual_seed(123)

import sys
sys.path.append("D:\llms_from_scratch\\")
sys.path.append("D:\llms_from_scratch\\chapter_5\\")

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry, format_style="alpaca")
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append([tokenizer.encode(instruction_plus_input), tokenizer.encode(full_text)])
            
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
            

file_path = "instruction-data.json"
with open(file_path, "r") as file:
    data = json.load(file)
    
end_of_text_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
# print("End of text token:", end_of_text_token)


def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    """
    The function adds a default <|endoftext|> token to the end of each sequence in the batch.
    Then the function pads the sequences to the length of the longest sequence in the batch.

    Args:
        batch (_type_): input_batch 
        pad_token_id (int, optional): token_id for the <|endoftext|> token. Defaults to 50256.
        device (str, optional): gpu or cpu. Defaults to "cpu".
    """
    batch_max_length = max(len(item)+1 for item in batch) # Get the length of the longest sequence in the batch.
    print("Batch max length: ", batch_max_length)
    inputs_list = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        import pdb; pdb.set_trace()
        
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        import pdb; pdb.set_trace()
        
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)
        
    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor


def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) # Shifted by one token. We add one token before so that this can work properly.
        inputs_list.append(inputs)
        targets_list.append(targets)
        
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor
    
    
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu",mask_instruction=True):
    batch_max_length = max(len(item[1])+1 for item in batch)
    inputs_list, targets_list = [], []
    # import pdb; pdb.set_trace()
    for b in batch:
        instruction = b[0]
        item = b[1]
        new_item = item.copy()  # Contains full_text = instruction (instruction + input) and the response.
        new_item += [pad_token_id]
        
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) 
        # I guess it's okay to have multiple end of text in the input only but not in the target
        # as we will see later.
        targets = torch.tensor(padded[1:]) # Shift by one token.
        
        mask = targets == pad_token_id # Get a boolean mask indicating the position where the token is the pad_token_id.
        indices = torch.nonzero(mask).squeeze() # Get the indices of the pad_token_id.
        if indices.numel() > 1: # Only remove pad_token_id from the target if there are more than one pad_token_id.
            targets[indices[1:]] = ignore_index # Ignore the pad_token_id, but keep the first pad_token_id (to denote end of text).
        
        # Truncate the inputs and targets if allowed_max_length is not None.    
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        # import pdb; pdb.set_trace() 
        # Mask instruction from the target.
        if mask_instruction:
            # Create a boolean array upto the length of the instruction.
            instruction_mask = torch.arange(len(targets), device=targets.device) < len(instruction)
            
            instruction_mask_indices = torch.nonzero(instruction_mask).squeeze()
            if instruction_mask_indices.numel() > 0: # Can be > 0 and not 1 unlike before.
                targets[instruction_mask_indices] = ignore_index
            
        # import pdb; pdb.set_trace()
        
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor
        
    


train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

train_dataset = InstructionDataset(train_data, tokenizer)
num_workers = 0
batch_size = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True
)

# Testing
# print("Train loader:")
# for inputs, targets in train_loader:
#     print("Inputs size: ", inputs.size())
#     print("Targets size: ", targets.size())
# batch = dataset[:3]
# # print("Batch: ", batch)
# result = custom_collate_fn(batch, pad_token_id=50256, device="cpu", mask_instruction=True)
# print("Result: ", result)


# The custom collate function sets the padding token to -100.
# We do this so that pytorch's cross entropy loss does not consider that token when calculating the loss.
# cross_entropy function has an ignore_index=-100 argument that takes care of this.
    
# # Testing the custom_collate_draft_1 function.
# inputs_1 = [0, 1, 2, 4, 3]
# inputs_2 = [0, 1, 2, 3]
# inputs_3 = [7, 8, 9]
# batch = [inputs_1, inputs_2, inputs_3]

# print("Batch before padding: ", batch)
# print("Batch after padding: ", custom_collate_fn(batch, pad_token_id=50256, device="cpu"))


from chapter_5.gpt_download import download_and_load_gpt2
from chapter_4.gpt_model import GPTModel
from chapter_5.loading_pretrained_weights import load_weights_into_gpt


BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate_att": 0.2,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.15,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[model_name])

print("New configuration: ", BASE_CONFIG)


model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
print("Model size: ", model_size)

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
# testing the pretrained model.
input_text = format_input(val_data[0])
print(input_text)

from chapter_5.utils import generate_text_simple, text_to_token_ids, token_ids_to_text, calc_loss_batch, calc_loss_loader
from chapter_5.utils import train_model_simple
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)
generated_text = token_ids_to_text(token_ids, tokenizer)
print("Generated text including instruction: ", generated_text)

response_text = generated_text[len(input_text):].strip()
print("Response text: ", response_text)

model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(
        model=model,
        data_loader=train_loader,
        device=device,
        num_batches=5 # Number of batches to calculate the loss (avg loss across batches).
    )
    val_loss = calc_loss_loader(
        model=model,
        data_loader=val_loader,
        device=device,
        num_batches=5
    )
    
print("Train loss: ", train_loss)
print("Validation loss: ", val_loss)


import time
start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=input_text,
    tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print("Training completed in {:.2f} minutes".format(execution_time_minutes))

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "gpt2-small-instruction-model.pth")
