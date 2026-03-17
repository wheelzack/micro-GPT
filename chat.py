#!/usr/bin/env python3
"""
chat.py - Interactive CLI chat interface for micro-GPT
Run: python chat.py --checkpoint checkpoints/model.pkl
"""

import numpy as np
import argparse
import pickle
import sys
from pathlib import Path


class ChatSession:
    def __init__(self, checkpoint_path, temperature=0.8, max_tokens=100, top_p=0.9):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Load checkpoint
        if not Path(checkpoint_path).exists():
            print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
            print("Train model first: python train.py")
            sys.exit(1)
        
        print(f"🔄 Loading model from {checkpoint_path}...")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.tokenizer = checkpoint['tokenizer']
        
        # Chat settings
        self.history = []
        self.system_prompt = "You are a helpful AI assistant. Answer concisely and accurately."
        
        print("✅ Model loaded successfully!")
        print(f"🌡️ Temperature: {temperature}")
        print(f"📏 Max tokens: {max_tokens}")
        print(f"🎯 Top-p: {top_p}\n")

    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def top_p_sampling(self, logits):
        """Nucleus (top-p) sampling"""
        # Sort logits
        sorted_logits = np.sort(logits)[::-1]
        sorted_indices = np.argsort(logits)[::-1]
        
        # Compute cumulative probabilities
        probs = self.softmax(sorted_logits / self.temperature)
        cum_probs = np.cumsum(probs)
        
        # Find cutoff
        cutoff_index = np.where(cum_probs > self.top_p)[0]
        if len(cutoff_index) > 0:
            cutoff_index = cutoff_index[0] + 1
        else:
            cutoff_index = len(probs)
        
        # Sample from top-p
        top_p_probs = probs[:cutoff_index]
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        top_p_indices = sorted_indices[:cutoff_index]
        
        chosen_idx = np.random.choice(cutoff_index, p=top_p_probs)
        return top_p_indices[chosen_idx]

    def sample_token(self, logits):
        """Sample next token with temperature and top-p"""
        if self.temperature == 0:
            return np.argmax(logits)
        
        return self.top_p_sampling(logits)

    def generate(self, prompt):
        """Generate response given prompt"""
        # Encode
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            return "I didn't understand that."
        
        generated = []
        
        for i in range(self.max_tokens):
            # Prepare context
            context = input_ids + generated
            
            # Forward pass through model
            try:
                # Try batch format first (your model might expect this)
                logits = self.model.forward(np.array([context]))
                if len(logits.shape) == 2:
                    logits = logits[0]
            except:
                # Fallback to single sequence
                logits = self.model.forward(context)
            
            # Get last position logits
            next_logits = logits[-1] if len(logits.shape) > 1 else logits
            
            # Sample
            next_token = self.sample_token(next_logits)
            generated.append(int(next_token))
            
            # Stop conditions
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Stop on repeated patterns (simple repetition check)
            if len(generated) > 10:
                last_5 = generated[-5:]
                prev_5 = generated[-10:-5]
                if last_5 == prev_5:
                    break
        
        # Decode response
        response = self.tokenizer.decode(generated)
        
        # Clean up
        response = response.replace("<|end|>", "").strip()
        response = response.replace("<|user|>", "").replace("<|assistant|>", "")
        
        return response

    def format_prompt(self, user_message):
        """Format conversation with system prompt and history"""
        prompt_parts = [f"<|system|>{self.system_prompt}<|end|>"]
        
        # Add recent history (last 3 exchanges for context)
        for exchange in self.history[-3:]:
            prompt_parts.append(f"<|user|>{exchange['user']}<|end|>")
            prompt_parts.append(f"<|assistant|>{exchange['bot']}<|end|>")
        
        # Add current message
        prompt_parts.append(f"<|user|>{user_message}<|end|>")
        prompt_parts.append("<|assistant|>")
        
        return "".join(prompt_parts)

    def print_help(self):
        """Show available commands"""
        print("""
📖 Commands:
  exit, quit, bye     - End chat
  clear, reset        - Clear conversation history
  temp=X              - Set temperature (0.1-2.0)
  topp=X              - Set top-p (0.1-1.0)
  max=X               - Set max tokens (10-500)
  system=X            - Change system prompt
  help                - Show this help
        """)

    def chat(self):
        """Main chat loop"""
        print("🤖 Welcome to micro-GPT Chat!")
        print("Type 'help' for commands, 'exit' to quit\n")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                cmd = user_input.lower()
                
                if cmd in ['exit', 'quit', 'bye', 'goodbye']:
                    print("👋 Goodbye!")
                    break
                
                elif cmd in ['clear', 'reset', 'new']:
                    self.history = []
                    print("🧹 Conversation history cleared!")
                    continue
                
                elif cmd == 'help':
                    self.print_help()
                    continue
                
                elif cmd.startswith('temp='):
                    try:
                        temp = float(cmd.split('=')[1])
                        self.temperature = max(0.1, min(2.0, temp))
                        print(f"🌡️ Temperature set to {self.temperature}")
                    except:
                        print("❌ Usage: temp=0.8")
                    continue
                
                elif cmd.startswith('topp='):
                    try:
                        top_p = float(cmd.split('=')[1])
                        self.top_p = max(0.1, min(1.0, top_p))
                        print(f"🎯 Top-p set to {self.top_p}")
                    except:
                        print("❌ Usage: topp=0.9")
                    continue
                
                elif cmd.startswith('max='):
                    try:
                        max_tok = int(cmd.split('=')[1])
                        self.max_tokens = max(10, min(500, max_tok))
                        print(f"📏 Max tokens set to {self.max_tokens}")
                    except:
                        print("❌ Usage: max=100")
                    continue
                
                elif cmd.startswith('system='):
                    self.system_prompt = user_input[7:]
                    print(f"🎭 System prompt updated!")
                    continue
                
                # Generate response
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
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Tip: Check if model and tokenizer are compatible")


def main():
    parser = argparse.ArgumentParser(
        description='Chat with micro-GPT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --checkpoint model.pkl
  python chat.py -c model.pkl -t 0.5 -m 50
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pkl file)'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.8,
        help='Sampling temperature (0.1-2.0, default: 0.8)'
    )
    
    parser.add_argument(
        '--max_tokens', '-m',
        type=int,
        default=100,
        help='Maximum tokens to generate (10-500, default: 100)'
    )
    
    parser.add_argument(
        '--top_p', '-p',
        type=float,
        default=0.9,
        help='Nucleus sampling parameter (0.1-1.0, default: 0.9)'
    )
    
    args = parser.parse_args()
    
    # Start chat session
    session = ChatSession(
        checkpoint_path=args.checkpoint,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    
    session.chat()


if __name__ == "__main__":
    main()
