#! python3
#%% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from preprocess import read_file, extract_unique_characters, create_character_mapping, split_dataset,encode,decode
from models import BigramModel, TransformerModel
from pathlib import Path
#%%
# get words
parentpath = Path(__file__).resolve().parent
filepath = parentpath / 'cheese.txt'
checkpointpath = parentpath / 'checkpointdir'
# create checkpoint dir if it doesn't exist
checkpointpath.mkdir(exist_ok=True)
text = read_file(filepath)
chars = extract_unique_characters(text)
stoi,itos = create_character_mapping(chars)
#%%
train_data, val_data = split_dataset(text,train_size=0.9)

# %% hyper parameters
batch_size = 4 # B
block_size = 8 # T 
num_heads = 6 # n_heads
n_embd = 36 # E
vocab_size = len(stoi) # V
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
sample_iters = 1000
n_layers = 3
#%% Get batch, estimate loss
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
 #%% Train model

model = TransformerModel(block_size,vocab_size,n_embd,num_heads,n_layers)
model.to(device)
optimizer = AdamW(model.parameters(),lr=0.001)
load_checkpoint = False
run_train = True
# if checkpoint.pth exists and user wants to load checkpoint
if checkpointpath / 'checkpoint.pth' and load_checkpoint:
    checkpoint = torch.load(checkpointpath/'checkpoint.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Checkpoint loaded')    
# training loop
#%% single forward pass 

xb,yb = get_batch('train')
logits,loss = model(xb,yb)

#%%
if run_train:
    for k in range(10000):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        # update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress
        if k % eval_iters == 0:
            losses = estimate_loss()
            print(f'iteration {k}, train loss: {losses["train"]:.3f}, val loss: {losses["val"]:.3f}')
        
        if k % sample_iters ==0:
            starting_phrase = 'To make swiss cheese'
            x_in = torch.tensor(encode(starting_phrase),dtype=torch.long,device=device).unsqueeze(0)
            sampling = model.generate(x_in,max_new_tokens=1000).squeeze()
            print(decode(sampling.tolist()))

    # at the end of the training save a checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpointpath/'checkpoint.pth')
