# 🔬 Sensora GPT-Atomic
**The most atomic, dependency-free Transformer implementation in Python.**

Developed by **Sensora Labs**, this repository contains a complete GPT-style architecture built from the ground up. We avoided heavy frameworks like PyTorch to demonstrate the raw mathematical beauty of Autograd and Self-Attention.

## 🚀 Key Features
- **Tensor-based Autograd**: A custom engine handling Matrix Multiplications ($A \times B$) and Backpropagation.
- **Karpathy-Inspired**: Optimized for clarity and "first-principles" understanding.
- **Pure NumPy**: Leveraging vectorized operations for 50x speed gains over scalar-based models.

## 🛠 Setup & Usage
1. Clone the repo: `git clone https://github.com/YourUser/sensora-gpt-atomic`
2. Install requirements: `pip install numpy`
3. Train the model: `python train.py`

## 🧠 Theory
The model uses a standard Transformer block consisting of:
1. **LayerNorm** for stability.
2. **Multi-Head Attention** for context awareness.
3. **GELU/ReLU** non-linearity.
