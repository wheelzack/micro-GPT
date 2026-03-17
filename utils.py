#!/usr/bin/env python3
"""
utils.py - Helper functions for micro-GPT
"""

import pickle


def save_checkpoint(model, tokenizer, filepath):
    """Save model and tokenizer together"""
    checkpoint = {
        'model': model,
        'tokenizer': tokenizer,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"💾 Saved to {filepath}")


def load_checkpoint(filepath):
    """Load model and tokenizer"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['model'], checkpoint['tokenizer']


def count_parameters(model):
    """Count trainable parameters"""
    total = 0
    for p in model.parameters():
        total += p.size
    return total


def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Quick text generation for testing"""
    import numpy as np
    
    input_ids = tokenizer.encode(prompt)
    generated = []
    
    for _ in range(max_tokens):
        # Forward
        logits = model.forward(input_ids + generated)
        next_logits = logits[-1]
        
        # Sample with temperature
        next_logits = next_logits / temperature
        probs = np.exp(next_logits - np.max(next_logits))
        probs = probs / np.sum(probs)
        
        next_token = np.random.choice(len(probs), p=probs)
        generated.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated)
