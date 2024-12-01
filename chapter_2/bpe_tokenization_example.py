from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))


# Instantiate a tiktoken tokenizer.
tokenizer = tiktoken.get_encoding('gpt2')

# The usage of this tokenizer is similar to the SimpleTokenizerV1 class.
text = "Hello, do you like tea? <|endoftext|> Himalayas are grinc."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Encodings: ", integers)


# Decode the integers.
decoded_text = tokenizer.decode(integers)
print("Decoded text: ", decoded_text)

# Playing with unknown words.
test_ids = tokenizer.encode("grinc")
for id in test_ids:
    print(tokenizer.decode([id]))