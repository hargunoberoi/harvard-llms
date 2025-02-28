#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class TransformerModel(nn.Module):

    def __init__(self, block_size,vocab_size,n_embd,n_heads,n_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # B,T --> B,T,E
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # B,T --> B,T,E
        self.blocks = nn.Sequential(*[AttentionBlock(block_size, n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # B,T,E --> B,T,V
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B,T = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            block_size = self.block_size
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


class AttentionBlock(nn.Module):
    def __init__(self, block_size, embd_size, num_heads):
        super().__init__()
        self.multi_head = CasualAttentionHead(block_size, embd_size,num_heads)
        self.feed_forward = FeedForward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4*embd_size),
            nn.ReLU(),
            nn.Linear(4*embd_size, embd_size),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)
        return x

class CasualAttentionHead(nn.Module):
    def __init__(self,block_size,embd_size,num_heads):
        super().__init__()
        # Initialize your layers here
        self.num_heads = num_heads
        self.key = nn.Linear(embd_size, embd_size, bias=False)
        self.query = nn.Linear(embd_size, embd_size, bias=False)
        self.value = nn.Linear(embd_size, embd_size, bias=False)
        self.proj = nn.Linear(embd_size, embd_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Define the forward pass here
        B,T,E = x.size()
        head_size = E // self.num_heads
        k = self.key(x).view(B,T,self.num_heads,-1).transpose(1,2)
        q = self.query(x).view(B,T,self.num_heads,-1).transpose(1,2)
        v = self.value(x).view(B,T,self.num_heads,-1).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1))* (head_size ** -0.5)
        masked_attn_scores = attn_scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn_probs = F.softmax(masked_attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1,2).contiguous().view(B,T,E)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MultiHead(nn.Module):
    def __init__(self,num_heads, block_size,embd_size):
        super().__init__()
        head_size = embd_size // num_heads
        self.heads = nn.ModuleList([Head(block_size,embd_size,head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd_size, embd_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # projection
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):
    def __init__(self,block_size,embd_size,head_size):
        super().__init__()
        # Initialize your layers here
    
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Define the forward pass here
        B,T,E = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2,-1))* (E ** -0.5)
        masked_attn_scores = attn_scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn_probs = F.softmax(masked_attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        return out

#%%
import json 
from pathlib import Path

if __name__ == "__main__":
    # load the config
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # load the tokenizer
    tokenizer_path = Path(__file__).parent / 'tokenizer.json'
    with open(tokenizer_path, 'r') as f:
        tokenizer = json.load(f)

    stoi = tokenizer
    itos = {v:k for k,v in stoi.items()}

    encode = lambda x: [stoi[c] for c in x]
    decode = lambda x: ''.join([itos[i] for i in x])

    # load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TransformerModel(**config)
    model.to(device)
    # load model weights which are consolidated.00.pth

    checkpoint_path = Path(__file__).parent / 'checkpointdir' / 'consolidated.00.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    starting_phrase = 'I urge you brother '
    x_in = torch.tensor(encode(starting_phrase),dtype=torch.long,device=device).unsqueeze(0)
    sampling = model.generate(x_in,max_new_tokens=1000).squeeze()
    print(decode(sampling.tolist()))
