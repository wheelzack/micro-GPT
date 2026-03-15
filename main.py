import sys
import argparse
from train import train_model
from generate import generate_text

def main():
    parser = argparse.ArgumentParser(description="Sensora Labs GPT CLI")
    parser.add_argument('mode', choices=['train', 'generate'], help="Choose 'train' or 'generate'")
    parser.add_argument('--text', type=str, default="sensora", help="Seed text for generation")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("🚀 Starting Sensora Labs Training Sequence...")
        train_model()
    elif args.mode == 'generate':
        print("🔮 Generating from Sensora Engine...")
        generate_text(args.text)

if __name__ == "__main__":
    main()
