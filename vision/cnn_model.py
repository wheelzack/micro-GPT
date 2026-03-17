#!/usr/bin/env python3
"""
vision/cnn_model.py - Complete CNN architectures
"""

import numpy as np
from .conv2d import Conv2D, ReLU, Flatten
from .pooling import MaxPool2D, GlobalAvgPool2D


class SimpleCNN:
    """
    Simple CNN for CIFAR-10 (32x32 images)
    Architecture: Conv-ReLU-Pool x3 -> Flatten -> FC
    """
    
    def __init__(self, num_classes=10, input_channels=3):
        # Block 1: 32x32 -> 16x16 (32 channels)
        self.conv1 = Conv2D(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        # Block 2: 16x16 -> 8x8 (64 channels)
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        # Block 3: 8x8 -> 4x4 (128 channels)
        self.conv3 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten: 128 * 4 * 4 = 2048
        self.flatten = Flatten()
        self.fc_input_size = 128 * 4 * 4
        
        # Classifier
        self.fc_weights = np.random.randn(self.fc_input_size, num_classes) * 0.01
        self.fc_bias = np.zeros(num_classes)
        
        # Gradients
        self.dW_fc = np.zeros_like(self.fc_weights)
        self.db_fc = np.zeros_like(self.fc_bias)
        
        self.cache = {}
    
    def forward(self, x):
        # Block 1
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # Block 2
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        # Block 3
        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        out = self.pool3.forward(out)
        
        # Flatten and classify
        out = self.flatten.forward(out)
        self.cache['fc_input'] = out
        
        logits = out @ self.fc_weights + self.fc_bias
        return logits
    
    def backward(self, dout):
        batch = dout.shape[0]
        
        # FC backward
        self.dW_fc = self.cache['fc_input'].T @ dout / batch
        self.db_fc = np.sum(dout, axis=0) / batch
        d_flatten = dout @ self.fc_weights.T
        
        # Backprop through blocks
        d_pool3 = self.flatten.backward(d_flatten)
        d_relu3 = self.relu3.backward(d_pool3)
        d_conv3 = self.conv3.backward(d_relu3)
        
        d_pool2 = self.pool2.backward(d_conv3)
        d_relu2 = self.relu2.backward(d_pool2)
        d_conv2 = self.conv2.backward(d_relu2)
        
        d_pool1 = self.pool1.backward(d_conv2)
        d_relu1 = self.relu1.backward(d_pool1)
        dx = self.conv1.backward(d_relu1)
        
        return dx
    
    def parameters(self):
        params = [
            self.conv1.weights, self.conv1.bias,
            self.conv2.weights, self.conv2.bias,
            self.conv3.weights, self.conv3.bias,
            self.fc_weights, self.fc_bias
        ]
        return params
    
    def zero_grad(self):
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self.conv3.zero_grad()
        self.dW_fc.fill(0)
        self.db_fc.fill(0)
    
    def save(self, filepath):
        """Save to NPZ"""
        np.savez(filepath,
                 conv1_w=self.conv1.weights, conv1_b=self.conv1.bias,
                 conv2_w=self.conv2.weights, conv2_b=self.conv2.bias,
                 conv3_w=self.conv3.weights, conv3_b=self.conv3.bias,
                 fc_w=self.fc_weights, fc_b=self.fc_bias)
    
    def load(self, filepath):
        """Load from NPZ"""
        data = np.load(filepath)
        self.conv1.weights = data['conv1_w']
        self.conv1.bias = data['conv1_b']
        self.conv2.weights = data['conv2_w']
        self.conv2.bias = data['conv2_b']
        self.conv3.weights = data['conv3_w']
        self.conv3.bias = data['conv3_b']
        self.fc_weights = data['fc_w']
        self.fc_bias = data['fc_b']


class ResidualBlock:
    """
    ResNet-style residual block
    Conv -> ReLU -> Conv -> (+ input) -> ReLU
    """
    
    def __init__(self, channels):
        self.conv1 = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        
        # Store input for skip connection
        self.x_cache = None
    
    def forward(self, x):
        self.x_cache = x
        
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        
        # Skip connection
        out = out + x
        out = self.relu2.forward(out)
        
        return out
    
    def backward(self, dout):
        # Backprop through second ReLU
        d_relu2 = self.relu2.backward(dout)
        
        # Gradient flows to both paths
        dx_skip = d_relu2  # Skip connection
        d_conv2 = self.conv2.backward(d_relu2)
        d_relu1 = self.relu1.backward(d_conv2)
        d_conv1 = self.conv1.backward(d_relu1)
        
        # Add gradients
        dx = d_conv1 + dx_skip
        
        return dx
    
    def parameters(self):
        return [self.conv1.weights, self.conv1.bias,
                self.conv2.weights, self.conv2.bias]
    
    def zero_grad(self):
        self.conv1.zero_grad()
        self.conv2.zero_grad()
