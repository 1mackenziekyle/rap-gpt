import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ===== Hyperparameters =====
batch_size = 64
block_size = 256
max_steps = 500
eval_interval = max_steps // 10
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_steps = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
train_split = 0.9
# -----


# ===== Program Operation Parameters =====
num_new_tokens = 10000
txt_file = 'kanye_lyrics.txt'
model_path = 'model_cp/gpt_50.pt'
do_training = False
print(f'Using device {device}')
print(f'Using text file {txt_file}')
# =====

# ====== Data ======
torch.manual_seed(1337)
with open(txt_file, encoding='unicode_escape') as f:
    text = f.read()
# tokenize
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mappings
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])



# ==== Split data into train and validation =====
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]



# ====== Batching Data ======
def get_batch(split):
    # generate  a small batch of data of inputs x and target y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)



# ===== Function to evaluate loss with less noise =====
@torch.no_grad() # tell torch not to store gradients
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out



# ===== Multi-headed attention =====
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



# ===== Transformer Block =====
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



# ===== Language Model =====
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, print_tokens=False):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if print_tokens:
                for i in range(10):
                    print('\n')
                print(decode(idx[0].tolist()))
                time.sleep(0.02)
        return idx



# ====== Instantiate =====
model = GPTLanguageModel()
if model_path is not None: # optional load checkpoint
    model.load_state_dict(torch.load(model_path))
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



# ====== Training ======
if do_training:
    print(f'Training GPT with {sum(p.numel() for p in m.parameters())/1e6:.1f}M parameters')
    start = time.time()
    for step in range(max_steps):
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f'step {step}: train loss {losses["train"]:.3f}, val loss {losses["val"]:.3f} | time elapsed {time.time()-start:.0f}s')
            torch.save(m.state_dict(), f'step{step}.pt')

        xb, yb = get_batch('train')
        # evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# ====== Generate ======
print("Generating Text...")
context = torch.zeros((1,1), dtype=torch.long, device=device)
m.generate(context, max_new_tokens=num_new_tokens, print_tokens=True)