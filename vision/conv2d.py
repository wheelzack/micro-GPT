#!/usr/bin/env python3
"""
vision/conv2d.py - 2D Convolution Layer (Pure NumPy)
"""

import numpy as np


class Conv2D:
    """
    2D Convolution layer with custom autograd support
    Optimized with Im2Col for faster matrix multiplication
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier/Glorot initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        self.weights = np.random.uniform(-limit, limit, 
            (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
        
        # Gradients
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        
        # Cache
        self.x = None
        self.x_cols = None
        
    def _im2col(self, x):
        """
        Convert image batches to column format for efficient convolution
        x: (batch, in_channels, height, width)
        returns: (batch * out_h * out_w, in_channels * k * k)
        """
        batch, channels, h, w = x.shape
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad if needed
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), 
                                  (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        # Create column matrix
        cols = np.zeros((batch, channels, self.kernel_size, self.kernel_size, 
                        out_h, out_w))
        
        for i in range(self.kernel_size):
            i_max = i + self.stride * out_h
            for j in range(self.kernel_size):
                j_max = j + self.stride * out_w
                cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:self.stride, j:j_max:self.stride]
        
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1)
        return cols, out_h, out_w
    
    def _col2im(self, cols, batch, h, w):
        """Reverse of im2col"""
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        cols_reshaped = cols.reshape(batch, out_h, out_w, self.in_channels, 
                                     self.kernel_size, self.kernel_size)
        cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)
        
        if self.padding > 0:
            h_padded, w_padded = h + 2 * self.padding, w + 2 * self.padding
        else:
            h_padded, w_padded = h, w
        
        dx_padded = np.zeros((batch, self.in_channels, h_padded, w_padded))
        
        for i in range(self.kernel_size):
            i_max = i + self.stride * out_h
            for j in range(self.kernel_size):
                j_max = j + self.stride * out_w
                dx_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] += cols_reshaped[:, :, i, j, :, :]
        
        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_padded
    
    def forward(self, x):
        """
        Forward pass using Im2Col + GEMM (faster)
        x: (batch, in_channels, height, width)
        returns: (batch, out_channels, out_h, out_w)
        """
        self.x = x
        batch, _, h, w = x.shape
        
        # Im2Col
        x_cols, out_h, out_w = self._im2col(x)
        self.x_cols = x_cols
        
        # Reshape weights for matrix multiplication
        # weights: (out_channels, in_channels, k, k) -> (out_channels, in_channels*k*k)
        w_rows = self.weights.reshape(self.out_channels, -1)
        
        # GEMM: (out_h*out_w*batch, in_channels*k*k) @ (in_channels*k*k, out_channels).T
        # Actually: (out_channels, in_channels*k*k) @ (in_channels*k*k, batch*out_h*out_w)
        out = w_rows @ x_cols.T  # (out_channels, batch*out_h*out_w)
        out = out.T  # (batch*out_h*out_w, out_channels)
        
        # Add bias and reshape
        out = out + self.bias  # Broadcasting
        out = out.reshape(batch, out_h, out_w, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (batch, out_channels, out_h, out_w)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass
        dout: (batch, out_channels, out_h, out_w)
        """
        batch, _, out_h, out_w = dout.shape
        
        # Reshape dout: (batch, out_channels, out_h, out_w) -> (batch*out_h*out_w, out_channels)
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Bias gradient
        self.db = np.sum(dout_reshaped, axis=0)
        
        # Weight gradient: dout_reshaped.T @ x_cols
        # (out_channels, batch*out_h*out_w) @ (batch*out_h*out_w, in_channels*k*k)
        self.dw = (dout_reshaped.T @ self.x_cols).reshape(self.weights.shape)
        
        # Input gradient
        w_rows = self.weights.reshape(self.out_channels, -1)
        dx_cols = dout_reshaped @ w_rows  # (batch*out_h*out_w, in_channels*k*k)
        
        # Col2Im
        _, h, w = self.x.shape
        dx = self._col2im(dx_cols, batch, h, w)
        
        return dx
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def zero_grad(self):
        self.dw.fill(0)
        self.db.fill(0)


class ReLU:
    """ReLU activation"""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask


class Flatten:
    """Flatten layer"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)
