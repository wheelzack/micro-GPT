#!/usr/bin/env python3
"""
vision/utils.py - Image preprocessing utilities
"""

import numpy as np


def load_image(path):
    """
    Load image from file (PNG/JPG)
    Returns: (height, width, channels) numpy array
    """
    try:
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except ImportError:
        print("PIL not available. Install: pip install Pillow")
        # Return dummy for testing
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


def preprocess_image(img, target_size=32):
    """
    Preprocess image for CNN
    img: (H, W, C) numpy array or PIL Image
    returns: (1, 3, target_size, target_size) normalized array
    """
    # Convert to numpy if PIL
    if hasattr(img, 'resize'):
        img = img.resize((target_size, target_size))
        img = np.array(img)
    
    # Resize if needed
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = _resize_image(img, target_size, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert HWC to CHW
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)  # (C, H, W)
    
    # Add batch dimension
    img = img[np.newaxis, :]  # (1, C, H, W)
    
    # Normalize with ImageNet stats (optional)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    
    return img


def _resize_image(img, new_h, new_w):
    """Simple bilinear resize"""
    old_h, old_w = img.shape[:2]
    
    row_scale = old_h / new_h
    col_scale = old_w / new_w
    
    row_idx = (np.arange(new_h) * row_scale).astype(int)
    col_idx = (np.arange(new_w) * col_scale).astype(int)
    
    return img[row_idx[:, None], col_idx[None, :]]


def augment_image(img):
    """
    Simple data augmentation
    img: (C, H, W)
    """
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = img[:, :, ::-1]
    
    # Random brightness
    factor = np.random.uniform(0.8, 1.2)
    img = img * factor
    
    # Clip
    img = np.clip(img, 0, 1)
    
    return img


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def predict_class(model, img_path, class_names=CIFAR10_CLASSES):
    """Predict class for single image"""
    img = load_image(img_path)
    img = preprocess_image(img, target_size=32)
    
    logits = model.forward(img)
    probs = _softmax(logits[0])
    
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    return class_names[pred_idx], confidence, probs


def _softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
