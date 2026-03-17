#!/usr/bin/env python3
"""
vision/pooling.py - Pooling layers
"""

import numpy as np


class MaxPool2D:
    """Max pooling with mask for backprop"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.max_indices = None
    
    def forward(self, x):
        self.x = x
        batch, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        out = np.zeros((batch, channels, out_h, out_w))
        self.max_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                patch = x[:, :, h_start:h_end, w_start:w_end]
                patch_reshaped = patch.reshape(batch, channels, -1)
                
                max_idx = np.argmax(patch_reshaped, axis=2)
                max_val = np.max(patch_reshaped, axis=2)
                
                out[:, :, i, j] = max_val
                
                # Store indices for backward
                max_h = max_idx // self.pool_size
                max_w = max_idx % self.pool_size
                self.max_indices[:, :, i, j, 0] = max_h
                self.max_indices[:, :, i, j, 1] = max_w
        
        return out
    
    def backward(self, dout):
        batch, channels, out_h, out_w = dout.shape
        dx = np.zeros_like(self.x)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                
                max_h = self.max_indices[:, :, i, j, 0]
                max_w = self.max_indices[:, :, i, j, 1]
                
                # Add gradient to max positions
                for b in range(batch):
                    for c in range(channels):
                        dx[b, c, h_start + max_h[b, c], w_start + max_w[b, c]] += dout[b, c, i, j]
        
        return dx


class AvgPool2D:
    """Average pooling"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
    
    def forward(self, x):
        self.x = x
        batch, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        out = np.zeros((batch, channels, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        return out
    
    def backward(self, dout):
        batch, channels, out_h, out_w = dout.shape
        dx = np.zeros_like(self.x)
        
        scale = 1.0 / (self.pool_size * self.pool_size)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                dx[:, :, h_start:h_end, w_start:w_end] += \
                    dout[:, :, i:i+1, j:j+1] * scale
        
        return dx


class GlobalAvgPool2D:
    """Global average pooling (used before classifier)"""
    
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x
        # Average over spatial dimensions
        return np.mean(x, axis=(2, 3))  # (batch, channels)
    
    def backward(self, dout):
        batch, channels = dout.shape
        h, w = self.x.shape[2], self.x.shape[3]
        
        # Distribute gradient evenly
        dx = dout.reshape(batch, channels, 1, 1) * np.ones((1, 1, h, w)) / (h * w)
        return dx
