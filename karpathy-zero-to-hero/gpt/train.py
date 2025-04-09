import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from model import LanguageModel

torch.manual_seed(42)
d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
device = next(device for device, available in d_opts if available)
print(f'using device: {device}')

with open('../data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f: data_txt = f.read()
# get all chars
chars = sorted(list(set(data_txt)))
vocab_size = len(chars)
# encode/decode funcs
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda e: "".join([itos[i] for i in e])
# encode data (chars to corresponding number)
data = torch.tensor(encode(data_txt), dtype=torch.long, device=device)
# split data into train/val
n = int(0.9*len(data)) # 90%, 10%
train_data = data[:n]
val_data = data[n:]

# hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_iters = 200
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = LanguageModel(block_size=block_size, n_embd=32, vocab_size=vocab_size, 
                      n_blocks=3, n_heads=4, device=device)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lossi = []

s_t = time.time()
for i in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    lossi.append(loss.item())
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {i}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
e_t = time.time()
print(f'training took: {e_t - s_t:.3f}s')

print('-- After Training')
print(f'loss: {lossi[-1]:.4f}')
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist()))