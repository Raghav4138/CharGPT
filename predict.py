import torch
from model import CharGPTLanguageModel
from utils import load_data, decode, create_vocabs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the vocabulary and initialize model
text = load_data('input.txt')
_, itos = create_vocabs(text)  # Only need 'itos' for decoding in predictions

vocab_size = len(itos)

model = CharGPTLanguageModel(vocab_size)
model.load_state_dict(torch.load('char_gpt_model.pth'))  # Load trained model weights
model.eval()

starting_index = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(idx=starting_index, max_tokens=1000)
print("-------------------------------------------------------")
print("GENERATED TEXT: ")

print(decode(generated_text.tolist()[0], itos))

