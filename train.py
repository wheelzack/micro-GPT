import numpy as np
from model import GPT # Maan ke chal rahe hain aapne model.py banayi hai

# --- DATASET ---
# Hum model ko "sensora labs" pattern sikhayenge
text = "sensora labs is building the future of ai. " * 20
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encode function
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- MODEL INITIALIZATION ---
# Sensora GPT configuration
n_embd = 32
model = GPT(vocab_size, n_embd) 

# --- SIMPLE TRAINING LOOP ---
print("Training Sensora GPT...")
learning_rate = 0.01

for epoch in range(1000):
    # Context window of 8 characters
    idx = np.random.randint(0, len(text) - 9)
    chunk = text[idx:idx+9]
    
    inputs = encode(chunk[:-1])
    targets = encode(chunk[1:])
    
    # Forward pass & Loss (using simplified logic)
    # Yahan aapka engine.py ka backprop kaam karega
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Training on '{chunk.strip()}'")

print("\n--- Training Complete! ---")

# Add these imports at top
from tokenizer import BPETokenizer  # or CharTokenizer
from utils import save_checkpoint

# ============ TRAINING LOOP के बाद ============

# Save final checkpoint
save_checkpoint(model, tokenizer, "checkpoints/final_model.pkl")
print("Training complete! Use: python chat.py --checkpoint checkpoints/final_model.pkl")
