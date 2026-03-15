import numpy as np

# --- SENSORA LABS CONFIG ---
class SensoraGPT:
    def __init__(self, vocab_size, n_embd=32, n_head=4, block_size=16):
        self.block_size = block_size
        self.n_embd = n_embd
        # Simple Weight Initialization
        self.wte = np.random.randn(vocab_size, n_embd) * 0.02 # token embeddings
        self.wpe = np.random.randn(block_size, n_embd) * 0.02 # position embeddings
        self.lm_head = np.random.randn(n_embd, vocab_size) * 0.02

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, idx):
        b, t = idx.shape
        tok_emb = self.wte[idx] # (b, t, n_embd)
        pos_emb = self.wpe[np.arange(t)] # (t, n_embd)
        x = tok_emb + pos_emb
        
        # Simplified Linear/Head projection for demo
        logits = x @ self.lm_head # (b, t, vocab_size)
        return logits

# --- TRAINING & DATA ---
text = "sensora labs is an ai research company. sensora labs builds gpt models."
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = SensoraGPT(vocab_size)
data = np.array(encode(text))

def train_step(model, x, y, lr=0.01):
    # Forward
    logits = model.forward(x)
    # Simple Cross-Entropy Loss Approximation
    probs = model.softmax(logits)
    
    # Logic for Backprop would go here using the 'Value' engine principles
    # For a repo, we focus on the structure.
    return np.mean(-np.log(probs[np.arange(len(y)), np.arange(len(y)), y]))

# --- RUNNING THE MODEL ---
print("Sensora Labs GPT Initialized.")
x_input = np.array([encode("sensora labs")])
logits = model.forward(x_input)
print(f"Input: 'sensora labs' | Output Shape: {logits.shape}")
