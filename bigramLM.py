import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# ------------

seed_val = 2810240409 # this is the time when i was writing this code (DDMMYYhhmm) so 28-10-2024, 04:09 in the morning
torch.manual_seed(seed_val)

# Now we read the input data, which is tiny-shakespheare data
with open('input.txt', 'r', encoding='utf-8') as input_file:
    text = input_file.read()

unique_chars = sorted((set(text)))
vocab_size = len(unique_chars)
stoi = {s: i for i, s in enumerate(unique_chars)}
itos = {i: s for i, s in enumerate(unique_chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(indexes):
    return "".join([itos[i] for i in indexes])

# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_indexes = torch.randint(low=0, high=len(data)-block_size, size=(batch_size, ))
    x_b = torch.stack([data[i:i+block_size] for i in random_indexes])
    y_b = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])
    return x_b , y_b

class BigramLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.ln_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) sized tensors where B is batch_dim and
        # T is the time dimention
        B, T = idx.shape
        tok_emb = self.model_embeddings(idx) # shape of this is (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.ln_head(x) # shape of this is (B,T,vocab_size)

        if targets==None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(idx) # (B, T, C)
            logits = logits[:, -1, :] # focus on last timestep (B, C)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim = 1) # (B, T + 1)
        return idx

m = BigramLM()

# Create a PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch('train')

    if iter % eval_iters == 0 :
        logits, loss = m(xb, yb)
        print(f'Loss after {iter} iterations is: train_loss: {loss.item()}')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

starting_index = torch.zeros((1, 1), dtype=torch.long)
generated_text = m.generate(idx=starting_index, max_tokens=300)
print("-------------------------------------------------------")
print("GENERATED TEXT: ")

print(decode(generated_text.tolist()[0]))