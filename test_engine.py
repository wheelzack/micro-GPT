import numpy as np
from engine import Tensor

def test_gradient_check():
    # 1. Setup a 1x1 Matrix for testing
    a_val, b_val = 2.0, 3.0
    a = Tensor([[a_val]])
    b = Tensor([[b_val]])
    
    # Forward pass: f = (a @ b) + (b @ b)
    f = (a @ b) + (b @ b) 
    f.backward()
    
    # 2. Manual Derivatives
    # f = a*b + b*b -> df/da = b, df/db = a + 2b
    expected_grad_a = [[3.0]]
    expected_grad_b = [[8.0]] # 2.0 + 2*(3.0)
    
    # 3. Validation
    assert np.allclose(a.grad, expected_grad_a), f"Grad A failed! Got {a.grad}"
    assert np.allclose(b.grad, expected_grad_b), f"Grad B failed! Got {b.grad}"
    
    print("✅ Sensora Engine: Gradient Check Passed!")

if __name__ == "__main__":
    test_gradient_check()
