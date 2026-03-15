# 🔬 Sensora GPT-Atomic
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![Build Status](https://github.com/wheelzack/micro-GPT/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

**The most atomic, dependency-free Transformer implementation in Python.**

---

## 🧠 Overview
This is a dependency-free, atomic implementation of a GPT transformer. Inspired by Andrej Karpathy's work, we've optimized this version for clarity and educational research at **Sensora Labs**. 

We don't use PyTorch or TensorFlow; we use pure math and NumPy to build the "brain" from scratch.

## 🚀 Features
- **Pure Python Autograd**: Custom-built engine to handle backpropagation.
- **Self-Attention**: Multi-head attention implementation from the ground up.
- **Dependency-Free**: Only requires `numpy`.
- **Automated Testing**: CI/CD pipeline integrated via GitHub Actions to verify gradients.

[Image of Transformer architecture diagram showing self-attention and feed-forward layers]

## 🛠️ Setup & Usage
```bash
# Clone the repository
git clone [https://github.com/wheelzack/micro-GPT.git](https://github.com/wheelzack/micro-GPT.git)
cd micro-GPT

# Install dependencies
pip install -r requirements.txt

# Start training
python train.py
