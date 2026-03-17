#!/usr/bin/env python3
"""
chat_vision.py - Multimodal chat with image understanding
"""

import numpy as np
import argparse
import pickle
from pathlib import Path

from vision import SimpleCNN, preprocess_image, CIFAR10_CLASSES


class VisionChat:
    """Chat with image + text capabilities"""
    
    def __init__(self, vision_checkpoint, text_checkpoint=None):
        # Load vision model
        print("🖼️ Loading vision model...")
        with open(vision_checkpoint, 'rb') as f:
            v_ckpt = pickle.load(f)
        self.vision_model = v_ckpt['model']
        
        # Load text model if available
        self.text_model = None
        if text_checkpoint and Path(text_checkpoint).exists():
            print("📝 Loading text model...")
            with open(text_checkpoint, 'rb') as f:
                t_ckpt = pickle.load(f)
            self.text_model = t_ckpt['model']
            self.tokenizer = t_ckpt['tokenizer']
        
        print("✅ Models loaded!\n")
    
    def analyze_image(self, img_path):
        """Analyze image and return description"""
        try:
            from PIL import Image
            img = Image.open(img_path)
        except:
            return "Could not load image"
        
        # Preprocess
        img_array = preprocess_image(img, target_size=32)
        
        # Predict
        logits = self.vision_model.forward(img_array)
        probs = self._softmax(logits[0])
        
        # Top 3 predictions
        top3 = np.argsort(probs)[-3:][::-1]
        
        result = "Image Analysis:\n"
        for i, idx in enumerate(top3):
            result += f"  {i+1}. {CIFAR10_CLASSES[idx]} ({probs[idx]:.1%})\n"
        
        return result
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def chat(self):
        """Interactive chat loop"""
        print("🤖 Vision-Enabled Chat")
        print("Commands:")
        print("  image <path>  - Analyze image")
        print("  text <msg>    - Chat with text")
        print("  exit          - Quit")
        print("-" * 40)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower().startswith('image '):
                img_path = user_input[6:].strip()
                result = self.analyze_image(img_path)
                print(f"🤖 Bot:\n{result}")
            
            elif user_input.lower().startswith('text '):
                if self.text_model:
                    text = user_input[5:]
                    # TODO: Integrate with text model
                    print(f"🤖 Bot: Text processing not yet integrated")
                else:
                    print("🤖 Bot: No text model loaded")
            
            else:
                # Try to detect if it's an image path
                if Path(user_input).exists() and user_input.endswith(('.png', '.jpg', '.jpeg')):
                    result = self.analyze_image(user_input)
                    print(f"🤖 Bot:\n{result}")
                else:
                    print("🤖 Bot: Please use 'image <path>' or 'text <message>'")


def main():
    parser = argparse.ArgumentParser(description='Vision-enabled chat')
    parser.add_argument('--vision', '-v', required=True, 
                       help='Path to vision checkpoint')
    parser.add_argument('--text', '-t', default=None,
                       help='Path to text/GPT checkpoint (optional)')
    
    args = parser.parse_args()
    
    chat = VisionChat(args.vision, args.text)
    chat.chat()


if __name__ == "__main__":
    main()
