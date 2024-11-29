import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
    
print("Total number of characters: ", len(raw_text))
print(raw_text[:100])


# Doing some regular expression magic.
import re
text = raw_text
result = re.split(r'(\s)', text)
# print(result)


# Accounting for word and puctuation characters.
result = re.split(r'([,.]|\s)', text)
# print(result)


# Further modifying the regular expression to account for different types of punctuations, such as question marks, exclamation marks, and colons.

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# print(result)
# print(len(result))



# Assigning token IDs to the words.


all_words = sorted(set(result))
vocab_size = len(all_words) # Unique words.
vocab = {token: integer for integer, token in enumerate(all_words)} # Just assigning a unique integer to each word by using the enumerate function.

# print(vocab)


