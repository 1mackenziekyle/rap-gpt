import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Hyperparameters =====
batch_size = 8
block_size = 4
max_steps = 10000
eval_interval = max_steps // 10
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_steps = 200
train_split = 0.9
print(f'Using device {device}')
# -----



# ====== Data ======
torch.manual_seed(1337)
with open('kanye_lyrics.txt') as f:
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




# ====== Model class ======
class BigramLangaugeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # idx is a (B,T) tensor of integers
            logits, loss = self(idx) # (B,T,C)
            # take only last character of each example
            logits = logits[:, -1, :] # Last characters (B,C)
            # generate probabilities along C dimension
            probs = F.softmax(logits, dim=-1) 
            # sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1) # (B,1)
            # append
            idx = torch.cat((idx, next_idx), dim=1)
        return idx



# ====== Instantiate =====
model = BigramLangaugeModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



# ====== Training ======
for step in range(max_steps):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {step}: train loss {losses["train"]:.3f}, val loss {losses["val"]:.3f}')

    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# ====== Generate ======
print("Bars incoming...")
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))