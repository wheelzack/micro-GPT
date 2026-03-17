#!/usr/bin/env python3
"""
chat.py - Interactive chat interface for micro-GPT
Usage: python chat.py --checkpoint model.pkl
"""

import numpy as np
import argparse
import pickle


class ChatSession:
    def __init__(self, checkpoint_path, temperature=0.8, max_tokens=100):
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"Loading model from {checkpoint_path}...")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.tokenizer = checkpoint['tokenizer']
        
        # Conversation history
        self.history = []
        self.system_prompt = "You are a helpful AI assistant."
        
        print("✅ Model loaded successfully!")
        print(f"Temperature: {temperature}, Max tokens: {max_tokens}\n")

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def sample(self, logits):
        """Sample with temperature"""
        if self.temperature == 0:
            return np.argmax(logits)
        
        logits = logits / self.temperature
        probs = self.softmax(logits)
        return np.random.choice(len(probs), p=probs)

    def generate(self, prompt):
        """Generate response"""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        
        generated = []
        
        for _ in range(self.max_tokens):
            # Forward pass - adapt based on your model's API
            context = input_ids + generated
            
            # Your model forward pass (modify as per your model.py)
            logits = self.model.forward(context)
            
            # Get last token logits
            next_logits = logits[-1] if len(logits.shape) > 1 else logits
            
            # Sample
            next_token = self.sample(next_logits)
            generated.append(next_token)
            
            # Stop on end token
            if next_token == self.tokenizer.eos_token_id:
                break
        
        # Decode
        response = self.tokenizer.decode(generated)
        return response.strip()

    def format_prompt(self, user_message):
        """Format with system and history"""
        prompt = f"<|system|>{self.system_prompt}<|end|>"
        
        # Add last 3 exchanges for context
        for h in self.history[-3:]:
            prompt += f"<|user|>{h['user']}<|end|>"
            prompt += f"<|assistant|>{h['bot']}<|end|>"
        
        prompt += f"<|user|>{user_message}<|end|><|assistant|>"
        return prompt

    def chat(self):
        print("🤖 micro-GPT Chat")
        print("Commands: 'exit' = quit, 'clear' = reset history, 'temp=X' = set temperature")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.history = []
                    print("🧹 History cleared!")
                    continue
                
                if user_input.startswith('temp='):
                    try:
                        self.temperature = float(user_input.split('=')[1])
                        print(f"🌡️ Temperature set to {self.temperature}")
                    except:
                        print("❌ Invalid temperature")
                    continue
                
                # Generate
                prompt = self.format_prompt(user_input)
                print("🤖 Bot: ", end="", flush=True)
                
                response = self.generate(prompt)
                print(response)
                
                # Save to history
                self.history.append({
                    'user': user_input,
                    'bot': response
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Chat with micro-GPT')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint (.pkl file)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                       help='Sampling temperature (0.1-1.5)')
    parser.add_argument('--max_tokens', '-m', type=int, default=100,
                       help='Max tokens to generate')
    
    args = parser.parse_args()
    
    session = ChatSession(
        checkpoint_path=args.checkpoint,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    session.chat()


if __name__ == "__main__":
    main()
