# 🚀 Micro-GPT (Sensora Labs Edition)

This is a dependency-free, atomic implementation of a GPT transformer. Inspired by Andrej Karpathy's work, we've optimized this version for clarity and educational research at **Sensora Labs**.

## 🧠 What's Inside?
- **Pure Python Autograd**: No PyTorch. No TensorFlow. Just math.
- **Character-level Training**: Trained to predict the next token in the "Sensora Labs" sequence.
- **Self-Attention**: Implementation of Multi-head attention from scratch.

## 🛠️ Setup
```bash
git clone [https://github.com/wheelzack/micro-GPT.git](https://github.com/wheelzack/micro-GPT.git)
pip install -r requirements.txt
python train.py


## 🧪 Proof of Concept (Inference)
After training for 5000 iterations on our custom dataset, the model can generate text based on the "Sensora" seed.

**Input:** `sens`
**Output:** `sensora labs is ai`

# 🔬 Sensora GPT-Atomic
![Build Status](https://github.com/wheelzack/micro-GPT/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

**The most atomic, dependency-free Transformer implementation in Python.**
