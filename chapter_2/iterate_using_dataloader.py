import tiktoken
from data_loading import GPTDatasetV1
from torch.utils.data import DataLoader


tokenizer = tiktoken.get_encoding('gpt2')

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers) # The drop_last=True will drop the last batch if it is not complete. This prevents loss spikes during training.
    return dataloader


with open('the-verdict.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    
dataloader = create_dataloader(text, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)  # Create an iterator from the dataloader.
first_batch = next(data_iter) # Get the first batch from the iterator.

print("First batch: ", first_batch)