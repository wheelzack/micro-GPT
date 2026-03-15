import numpy as np
from engine import Tensor

def test_gradient_check():
    # 1. Setup a simple operation: f(a, b) = (a * b) + b^2
    a_val, b_val = 2.0, 3.0
    a = Tensor([a_val])
    b = Tensor([b_val])
    
    # Forward pass
    # f = a*b + b*b = 2*3 + 3*3 = 15
    f = (a @ b) + (b @ b) 
    f.backward()
    
    # 2. Manual Derivatives (Calculus)
    # df/da = b = 3.0
    # df/db = a + 2b = 2 + 2(3) = 8.0
    
    expected_grad_a = 3.0
    expected_grad_b = 8.0
    
    # 3. Validation
    assert np.allclose(a.grad, expected_grad_a), f"Grad A failed! Expected {expected_grad_a}, got {a.grad}"
    assert np.allclose(b.grad, expected_grad_b), f"Grad B failed! Expected {expected_grad_b}, got {b.grad}"
    
    print("✅ Sensora Engine: Gradient Check Passed!")

if __name__ == "__main__":
    test_gradient_check()
