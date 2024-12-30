import torch
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