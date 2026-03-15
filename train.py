import numpy as np
from engine import Tensor
from gpt import MicroGPT

# 1. Data Preparation
text = "sensora labs " * 50  # Synthetic dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 2. Initialize Sensora GPT
model = MicroGPT(vocab_size=vocab_size, n_embd=16)
learning_rate = 0.1

print("--- Starting Sensora Labs Training ---")

# 3. Training Loop
for step in range(500):
    # Grab a small chunk of data
    raw_inputs = text[step % 10 : (step % 10) + 4]
    raw_targets = text[(step % 10) + 1 : (step % 10) + 5]
    
    inputs = [stoi[c] for c in raw_inputs]
    targets = [stoi[c] for c in raw_targets]

    # Forward Pass
    logits = model.forward(inputs) # Tensor object
    
    # Simplified Loss: MSE toward target embedding (for atomic demonstration)
    # In a full GPT, this would be Cross-Entropy
    target_vals = np.zeros_like(logits.data)
    for i, t_idx in enumerate(targets):
        target_vals[i, t_idx % 16] = 1 # Simplified target mapping
    
    diff = logits.data - target_vals
    loss_val = np.mean(diff**2)
    
    # Backward Pass (The Autograd Magic)
    logits.grad = 2 * diff / diff.size # Manual seed for MSE grad
    logits.backward()
    
    # Update Weights (Optimization)
    model.wte.data -= learning_rate * model.wte.grad
    model.wout.data -= learning_rate * model.wout.grad
    
    # Zero Grads for next step
    model.wte.grad = np.zeros_like(model.wte.data)
    model.wout.grad = np.zeros_like(model.wout.data)

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss_val:.4f}")

print("Training Complete. Model 'Sensora-v1' is ready.")
