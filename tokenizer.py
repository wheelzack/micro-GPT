#!/usr/bin/env python3
"""
tokenizer.py - Byte Pair Encoding Tokenizer for micro-GPT
"""

import pickle
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}  # id -> bytes
        self.merges = {}  # (int, int) -> int
        
        # Special tokens
        self.special_tokens = {
            "<|pad|>": 0,
            "<|end|>": 1,
            "<|unk|>": 2,
            "<|system|>": 3,
            "<|user|>": 4,
            "<|assistant|>": 5,
        }
        
        # Initialize vocab with special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode('utf-8')
        
        self.next_id = len(self.special_tokens)
        
        # Quick access
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def get_stats(self, ids):
        pairs = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            pairs[pair] += 1
        return pairs

    def merge(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, verbose=False):
        print(f"Training BPE with vocab_size={self.vocab_size}")
        
        # Start with raw bytes
        ids = list(text.encode('utf-8'))
        num_merges = self.vocab_size - len(self.special_tokens)
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            new_id = self.next_id
            self.next_id += 1
            
            self.merges[pair] = new_id
            
            # Add to vocab
            byte0 = self.vocab.get(pair[0], bytes([pair[0]]))
            byte1 = self.vocab.get(pair[1], bytes([pair[1]]))
            self.vocab[new_id] = byte0 + byte1
            
            ids = self.merge(ids, pair, new_id)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}/{num_merges}: pair={pair}, freq={stats[pair]}")
        
        print(f"Training complete! Vocab size: {len(self.vocab)}")
        return self

    def encode(self, text):
        # Handle special tokens
        for token_str, token_id in self.special_tokens.items():
            text = text.replace(token_str, f"\0{token_id}\0")
        
        parts = text.split("\0")
        final_ids = []
        
        for part in parts:
            if part.isdigit() and int(part) in self.special_tokens.values():
                final_ids.append(int(part))
            elif part:
                # Encode as bytes
                ids = list(part.encode('utf-8'))
                for (p0, p1), new_id in self.merges.items():
                    ids = self.merge(ids, (p0, p1), new_id)
                final_ids.extend(ids)
        
        return final_ids

    def decode(self, ids):
        bytes_list = []
        for idx in ids:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            else:
                bytes_list.append(b'')
        
        try:
            return b''.join(bytes_list).decode('utf-8', errors='ignore')
        except:
            return ""

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'vocab_size': self.vocab_size,
                'next_id': self.next_id
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']
        self.next_id = data['next_id']
        
        # Update references
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        return self


# Simple char tokenizer for quick testing
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.eos_token_id = 0
        self.pad_token_id = 0
        
    def train(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i+1 for i, c in enumerate(chars)}
        self.id_to_char = {i+1: c for i, c in enumerate(chars)}
        return self
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'char_to_id': self.char_to_id, 'id_to_char': self.id_to_char}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = data['id_to_char']
        return self
