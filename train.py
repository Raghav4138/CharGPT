import torch
from model import CharGPTLanguageModel
from torch.nn import functional as F
from utils import load_data, create_vocabs, encode

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
seed_val = 3110241528
# ------------

torch.manual_seed(seed_val)

# Now we read the input data, which is tiny-shakespeare data by default, (or you can use your any other text.)
text = load_data('input.txt')
stoi, itos = create_vocabs(text)

# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text, stoi), dtype=torch.long)
train_size = int(0.9*len(data))  # first 90% will be train data, rest val
train_data = data[:train_size]
val_data = data[train_size:]

vocab_size = len(stoi)

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


model = CharGPTLanguageModel(vocab_size)
m = model.to(device)

print(f"{sum(p.numel() for p in m.parameters())/1e6:.4f}M Training Parameters")
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

torch.save(model.state_dict(), 'char_gpt_model.pth')