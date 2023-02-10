import torch
import torch.nn as nn
from torch.nn import functional as F

# Set the random seed for reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------------

# Read the input text
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(str): return [stoi[c] for c in str]
def decode(vec): return ''.join([itos[i] for i in vec])

# Split the data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Random sampling
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x.to(device), y.to(device)

# Language Model
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)  # (B, T, C)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)  # POTENTIAL BUG – ICONSISTENT SHAPE!
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # Run the neural network
      logits, loss = self(idx) # logits: B, T, C; loss: not used
      # For each batch, take the very last prediction for each time series
      logits = logits[:, -1, :]  # B, C
      # Compute probabilities via softmax
      probs = torch.softmax(logits, dim=1) # B, C
      # Compute the indices of the actual characters
      idx_next = torch.multinomial(probs, num_samples=1) # B, 1
      # Add the next indices to the originals
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Training
m = BigramLanguageModel(vocab_size)
m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for steps in range(max_iters):
  # Sample data
  xb, yb = get_batch('train')

  # Evaluate the loss funcwtion
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
