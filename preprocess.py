# imports
import torch

#%% get words

def read_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text

text = read_file('cheese.txt')

#%%
def extract_unique_characters(text):
    characters = sorted(set(text))
    return characters

chars = extract_unique_characters(text)
vocab_size = len(chars)

def create_character_mapping(characters):
    stoi = {s:i for i,s in enumerate(characters)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

stoi, itos = create_character_mapping(chars)
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x]) 
#%% get train and validation set from the dataset
def split_dataset(text,train_size=0.8):
    data_tensor = torch.tensor(encode(text))
    split = int(data_tensor.shape[0]*(train_size))
    return data_tensor[:split], data_tensor[split:]
