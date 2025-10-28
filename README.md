# Learning to Code by trying to replicate tinygrad from scratch

### Example

```python
import numpy as np
from tinygrad.tensor import Tensor

x = Tensor(np.eye(3))
y = Tensor(np.array([[2, 0, 0, -2, 0]]))
z = y.dot(x).sum()
z.backward()

print(x.grad)
print(y.grad)
```
### Basic MNIST code is working
### [Video](https://www.youtube.com/watch?v=Xtws3-Pk69o)
