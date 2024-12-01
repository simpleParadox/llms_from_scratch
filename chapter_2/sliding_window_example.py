import tiktoken
from importlib.metadata import version
print("tiktoken version: ", version("tiktoken"))
tokenizer = tiktoken.get_encoding('gpt2')
with open('the-verdict.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    
enc_text = tokenizer.encode(text)
print("Length of the encoded text: ", len(enc_text))


# Get a sample of the encoded text which contains the text without the first 50 tokens.
sample_text = enc_text[50:]


# Now we created some example input-target pairs.
# We will use a sliding window to create input-target pairs.
context_size = 6
x = sample_text[:context_size]
y = sample_text[1: context_size + 1]
print(f'x: {x}')
print(f'y:      {y}')


# Ideally, we want to create many pairs of input-target pairs.
for i in range(1, context_size+1):
    context = sample_text[:i]
    target = sample_text[i]
    print(f'context: {tokenizer.decode(context)} ---> target: {tokenizer.decode([target])}')