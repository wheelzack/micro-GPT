import numpy as np
from model import GPT

def load_and_chat():
    # Load weights from your training run
    # (Ensure you save your weights as 'weights.npy' during training)
    print("🧠 Sensora Labs GPT is loading...")
    
    # Simple loop for interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        # Here we would call our generate function
        # response = model.generate(user_input)
        print(f"Sensora: [Model is processing pattern for '{user_input}']...")

if __name__ == "__main__":
    load_and_chat()
