import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
max_iters = 5000  # iterations for train loop
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embed = 192
n_layer = 3  # number of transformer layers in the model
n_head = 3  # number of self attention head in a single transformer block
dropout = 0.2
# ------------

seed_val = 3110240020  # this is the time when i was writing this code (DDMMYYhhmm) so 28-10-2024, 04:09 in the morning
torch.manual_seed(seed_val)

# Now we read the input data, which is tiny-shakespeare data by default, (or you can use your any other text.)
with open('input.txt', 'r', encoding='utf-8') as input_file:
    text = input_file.read()

unique_chars = sorted((set(text)))
vocab_size = len(unique_chars)
stoi = {s: i for i, s in enumerate(unique_chars)}
itos = {i: s for i, s in enumerate(unique_chars)}

def encode(text):
    # takes a string and returns a list of integers (indexes)
    return [stoi[c] for c in text]

def decode(indexes):
    # takes a list of indexes and returns a decoded text
    return "".join([itos[i] for i in indexes])


# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(0.9*len(data))  # first 90% will be train data, rest val
train_data = data[:train_size]
val_data = data[train_size:]


# Data Loader
def get_batch(split):
    """ Generates a single batch of data and targets, to be fed into the model. """

    data = train_data if split == 'train' else val_data
    random_indexes = torch.randint(low=0, high=len(data)-block_size, size=(batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in random_indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """ Smoothens the loss value over 'eval_iters' iterations """
    """ Returns an averaged loss """

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


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)  # (C, H)
        self.query = nn.Linear(n_embed, head_size, bias=False)  # (C, H)
        self.value = nn.Linear(n_embed, head_size, bias=False)  # (C, H)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inputs of size (batch_size, time_dim, channels)
        # return outputs of size (batch_size, time_dim, head_size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)

        # Computes the self attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, H) @ (B, H, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Computes the weighted aggregation of the values
        v = self.value(x) # (B, T, C) * (C, H) --> (B, T, H)
        out = wei @ v # (B, T, T) @ (B, T, H) --> (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attentions concatenated together"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """a simple MLP linear layer followed by a non_linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer Block: communication followed by communication """

    def __init__(self, n_embed, n_head):
        # n_embed: embeddings size, n_head: the number of attention heads per block
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Here we are also adding Residual Connections that help NNs train better
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Char_GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed) # final layer norm layer
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) sized tensors where B is batch_dim and
        # T is the time dimension
        B, T = idx.shape

        tok_emb = self.model_embeddings(idx) # shape of this is (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # shape of this is (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) # (B, T, C)
            logits = logits[:, -1, :] # focus on last timestep (B, C)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim = 1) # (B, T + 1)
        return idx


model = Char_GPTLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(f"Device: {device}")

# Create a PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch('train')

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train_loss: {losses['train']:.4f}, val_loss {losses['val']:.4f}")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

starting_index = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(idx=starting_index, max_tokens=300)
print("-------------------------------------------------------")
print("GENERATED TEXT: ")

print(decode(generated_text.tolist()[0]))