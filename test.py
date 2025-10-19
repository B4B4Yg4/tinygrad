import numpy as np
from tensor import Tensor


x = np.random.randn(1, 3)
W = np.random.randn(3, 3)
out = x.dot(W)
print(out)
