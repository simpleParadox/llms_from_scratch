
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import sys
import json
sys.path.append("D:\llms_from_scratch\\")
from chapter_4.gpt_model import GPTModel
from chapter_2.iterate_using_dataloader import create_dataloader 
import tiktoken
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding('gpt2')
file_path = '..\\the-verdict.txt'
with open(file_path, "r") as f:
    text = f.read()
    
total_characters = len(text)
total_tokens = len(tokenizer.encode(text))
print("Total characters: ", total_characters)
print("Total tokens: ", total_tokens)

with open('D:\llms_from_scratch\chapter_4\gpt_config.json', 'r') as f:
    cfg = json.load(f)
cfg = cfg['GPT_CONFIG_124M']
# Create train and test splits.
train_ratio = 0.90
split_idx = int(train_ratio * len(text)) # Take the first 90% of the text for training.
train_text = text[:split_idx]
test_text = text[split_idx:]

# Internally uses tiktoken gpt2.
train_loader = create_dataloader(
    train_text, batch_size=2,
    max_length=cfg["context_length"],
    stride=cfg["context_length"],
    drop_last=True,
    shuffle=True
)

test_loader = create_dataloader(
    test_text, batch_size=2,
    max_length=cfg["context_length"],
    stride=cfg["context_length"],
    drop_last=True,
    shuffle=True
)

# Check if everything is correct.
# for x, y in train_loader:
#     print(x.shape, y.shape)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add the batch dimension.
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    
    # Make sure you understand the shape conversion.
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1)
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
    
    
# Let's test the loss calculation functions.
#  with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device, num_batches=2)
#     test_loss = calc_loss_loader(test_loader, model, device, num_batches=2)

# print("Train loss: ", train_loss)
# print("Test loss: ", test_loss)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device=device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device=device, num_batches=eval_iter
        )
    
    # Switch back to training mode.
    model.train()
    return train_loss, val_loss

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=5,
                         top_k=3, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # only get the last context_size tokens.
        with torch.no_grad():
            logits = model(idx_cond)
            
        # Get the last token from the model output. This will be used to get the probability of the next token.
        logits = logits[:, -1, :] # Get the last token.
        
        # NOTE: Doing greedy decoding.
        # import pdb; pdb.set_trace()
        
        # NOTE: Doing probabilistic sampling.
        # torch.manual_seed(123)
        
        if top_k > 0:
            # Do top-k sampling.
            top_logits, top_indices = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] # Get the minimum value of the top-k logits for each sample.
            logits = torch.where(
                logits < min_val, # Condition.
                torch.tensor(float('-inf')).to(logits.device), # Return this value if the condition is True.
                logits # Return the original value if the condition is False.
            )
            
        if temperature > 0.0: # Scaling with temperature.
            logits = logits / temperature # Higher temperature will make the distribution more uniform.
            # Lower temperature will make the distribution more peaked (more greedy).
            # Also, using the temperature of 1 means that the distribution is not changed
            # and the torch.multinomial will use the original logit values.
            probas = torch.softmax(logits, dim=-1)
            # torch.multinomial requires that the inputs are non-negative - hence getting the softmax.
            idx_next = torch.multinomial(probas, num_samples=1) # Get one sample.
            
            # NOTE: In top-k sampling, you only consider the top-k most likely tokens.
            # We use an -inf mask to zero out the logits of the tokens that are not in the top-k.
            # We can simply use the torch.topk function to get the top-k values.
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Get the index of the token with the highest probability.
            
        if idx_next == eos_id:
            break
        
        # Append the new token to the input sequence.
        idx = torch.cat((idx, idx_next), dim=-1) # Concatenate the new token to the input sequence.
    
    return idx # Return the input sequence with the new tokens appended.


def generate_and_print_sample(model, tokenizer, device, start_context, temperature):
    model.eval()
    context_size = model.pos_emb.weight.shape[0] # Get the context size from the positional embeddings.
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model, encoded, max_new_tokens=20, context_size=context_size, temperature=temperature
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    
    # Switch back to training mode.
    model.train()
    

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, temperature=0.0):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Evaluate every eval_freq steps. 
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch: {epoch}, Global step: {global_step}, "
                    f"Train loss: {train_loss}, Val loss: {val_loss}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context, temperature=temperature
        )
        
    # Print gradient means.
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         print(f"{name} has gradient mean of {param.grad.mean()} and std of {param.grad.std()}")
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label='Train loss', color='b')
    ax1.plot(epochs_seen, val_losses, label='Val loss', color='r', linestyle='-.')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() # Create a second x-axis. This will be plotted on the top of the plot.
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()

model = GPTModel(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, test_loader,
    optimizer=optimizer, device=device,
    num_epochs=num_epochs, eval_freq=10, eval_iter=10,
    start_context="Every effort moves you", tokenizer=tokenizer, temperature=0.5
)

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# print("Printing gradient means after training.")
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print(f"{name} has gradient mean of {param.grad.mean()} and std of {param.grad.std()}")
# Different decoding strategies to control  randomness of the generated text.
model.to('cpu')
model.eval()
token_ids = generate_text_simple(
    model, text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=20, context_size=cfg["context_length"]
)
print("Output text: ", token_ids_to_text(token_ids, tokenizer))

# Save the model and the optimizer.
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "model_and_optimizer.pth")

# Load the model and the optimizer.
# NOTE: This step will take a lot of time and possibly make the computer slow.
# Therefore, I'm commenting out the following lines.
# model = GPTModel(cfg)
# checkpoint = torch.load("model_and_optimizer.pth")
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()
# print("Model loaded successfully.")