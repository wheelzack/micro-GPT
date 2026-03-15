import numpy as np

class DataLoader:
    def __init__(self, filepath, block_size):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = f.read()
        
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size

    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, l): return ''.join([self.itos[i] for i in l])

    def get_batch(self, batch_size):
        # Generate random starting points in the text
        ix = np.random.randint(0, len(self.data) - self.block_size, batch_size)
        x = np.array([self.encode(self.data[i:i+self.block_size]) for i in ix])
        y = np.array([self.encode(self.data[i+1:i+self.block_size+1]) for i in ix])
        return x, y
