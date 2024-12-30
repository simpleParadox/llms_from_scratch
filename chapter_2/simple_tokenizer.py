# Implementing a simple tokenizer from scratch.
from chapter_2_tokenizing_text import vocab, all_words
import re
class SimpleTokenizerV1:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary # This is mapping from the word to the integer.
        self.int_to_str = {v: k for k, v in vocabulary.items()}
        
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # First we need to check if the word is in the vocabulary. If it's not, we will assign the <UNK> token to it.
        preprocessed = [word if word in self.str_to_int else "<UNK>" for word in preprocessed]
        
        ids = [self.str_to_int[word] for word in preprocessed]  # Get the mappings from the vocabulary.
        
        return ids
    
    def decode(self, ids): 
        text = " ".join([self.int_to_str[i] for i in ids]) # Join the words after obtaining them from the int_to_str mappings.
        
        # Now when the text is joined, there will be spaces before the punctuations. We can remove them by using the following regular expression.
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text) # Essentially a substitution of the space before the punctuation with the punctuation itself.
        return text 




tokenizer = SimpleTokenizerV1(vocab)
text = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

# NOTE: The tokenizer will work with the vocabulary that's only in the the-verdict.txt file.
# To handle out of vocabulary words, we need to extend the vacabulary with special tokens, such as <UNK> for unknown words.
# We are also going to add <|endoftext|> to indicate the end of the text. This is a common practice in language modeling and acts as a separator for two 
# unrelated pieces of text.

# Let's modify the tokenizer to handle these special tokens.
all_words.extend(["<|endoftext|>", "<UNK>"])

# Create new vocabulary with the extended tokens. This is necessary to 
# handle out of vocabulary words.
vocab = {token: integer for integer, token in enumerate(all_words)}
print("Length of the new vocabulary: ", len(vocab))



# Let's test the new tokenizer.
text1 = "Hello, do you like  tea?"
text2 = "Himalayas are beautiful."
text = "<|endoftext|> ".join([text1, text2])
print(text)

tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))