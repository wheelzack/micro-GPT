#!/usr/bin/env python3
"""
train_vision.py - Train CNN on image dataset
"""

import numpy as np
import pickle
from pathlib import Path
import sys

from vision import SimpleCNN


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(logits, labels):
    """Cross entropy loss with gradients"""
    batch_size = logits.shape[0]
    probs = softmax(logits)
    
    # Negative log likelihood
    correct_logprobs = -np.log(probs[range(batch_size), labels] + 1e-8)
    loss = np.mean(correct_logprobs)
    
    # Gradient
    dlogits = probs.copy()
    dlogits[range(batch_size), labels] -= 1
    dlogits /= batch_size
    
    return loss, dlogits


def sgd_update(model, lr):
    """Simple SGD update"""
    model.conv1.weights -= lr * model.conv1.dw
    model.conv1.bias -= lr * model.conv1.db
    model.conv2.weights -= lr * model.conv2.dw
    model.conv2.bias -= lr * model.conv2.db
    model.conv3.weights -= lr * model.conv3.dw
    model.conv3.bias -= lr * model.conv3.db
    model.fc_weights -= lr * model.dW_fc
    model.fc_bias -= lr * model.db_fc


def train_epoch(model, X, y, batch_size=32, lr=0.01):
    """Train one epoch"""
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        # Forward
        logits = model.forward(X_batch)
        loss, dlogits = cross_entropy_loss(logits, y_batch)
        
        # Backward
        model.zero_grad()
        model.backward(dlogits)
        
        # Update
        sgd_update(model, lr)
        
        total_loss += loss
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"  Batch {num_batches}/{num_samples//batch_size}, Loss: {loss:.4f}")
    
    return total_loss / num_batches


def accuracy(model, X, y, batch_size=32):
    """Calculate accuracy"""
    correct = 0
    total = 0
    
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        logits = model.forward(X_batch)
        preds = np.argmax(logits, axis=1)
        
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    
    return correct / total


def generate_dummy_data(num_train=1000, num_test=200):
    """Generate dummy CIFAR-10-like data for testing"""
    print("Generating dummy CIFAR-10 data...")
    X_train = np.random.randn(num_train, 3, 32, 32) * 0.1
    y_train = np.random.randint(0, 10, num_train)
    
    X_test = np.random.randn(num_test, 3, 32, 32) * 0.1
    y_test = np.random.randint(0, 10, num_test)
    
    return X_train, y_train, X_test, y_test


def main():
    print("🖼️ Sensora-Vision CNN Training")
    print("=" * 50)
    
    # Setup
    Path("checkpoints/vision").mkdir(parents=True, exist_ok=True)
    
    # Data
    X_train, y_train, X_test, y_test = generate_dummy_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Model
    model = SimpleCNN(num_classes=10, input_channels=3)
    num_params = sum(p.size for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Training
    epochs = 20
    lr = 0.01
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, X_train, y_train, batch_size=32, lr=lr)
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        
        print(f"  Loss: {train_loss:.4f} | Train: {train_acc:.2%} | Test: {test_acc:.2%}")
        
        # Save
        if (epoch + 1) % 5 == 0:
            model.save(f"checkpoints/vision/cnn_epoch_{epoch+1}.npz")
        
        lr *= 0.95  # Decay
    
    # Save final
    model.save("checkpoints/vision/cnn_final.npz")
    
    # Also save as pickle for chat interface
    checkpoint = {
        'model': model,
        'tokenizer': None,
        'model_type': 'vision'
    }
    with open("checkpoints/vision/final.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print("\n✅ Training complete!")
    print("Test: python vision/utils.py with your image")


if __name__ == "__main__":
    main()
