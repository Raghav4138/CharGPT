def load_data(file_path='input.txt'):
    # Now we read the input data, which is tiny-shakespeare data by default, (or you can use your any other text.)
    with open('input.txt', 'r', encoding='utf-8') as input_file:
        text = input_file.read()
    return text


def create_vocabs(text):
    unique_chars = sorted((set(text)))
    vocab_size = len(unique_chars)
    stoi = {s: i for i, s in enumerate(unique_chars)}
    itos = {i: s for i, s in enumerate(unique_chars)}
    return stoi, itos


def encode(text, stoi):
    # takes a string and returns a list of integers (indexes)
    return [stoi[c] for c in text]


def decode(indexes, itos):
    # takes a list of indexes and returns a decoded text
    return "".join([itos[i] for i in indexes])
