import numpy as np
from model import GPT # Aapka modular model file

def generate(model, start_text, length=20):
    model.eval() # Model ko inference mode mein dalna
    context = encode(start_text)
    
    print(f"Seed: {start_text}")
    print("Generating: ", end="")
    
    for _ in range(length):
        # Context ko model mein feed karna
        idx_cond = np.array(context[-8:]) # Context window limit
        logits = model.forward(idx_cond)
        
        # Last character ke logits nikalna
        last_logit = logits[-1, :]
        
        # Probabilities nikalne ke liye Softmax
        probs = np.exp(last_logit) / np.sum(np.exp(last_logit))
        
        # Agla character sample karna
        next_char_idx = np.random.choice(len(probs), p=probs)
        
        # Result ko update karna
        context.append(next_char_idx)
        print(itos[next_char_idx], end="", flush=True)
    print("\n")

# Use case:
# generate(model, "sens", length=15)
