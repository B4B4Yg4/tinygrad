import numpy as np
from tensor import Tensor


def test_tinygrad():

    x = Tensor(x_init)
    W = Tensor(W_init)
    m = Tensor(m_init)

    out = x.dot(W)
    outr = out.relu()
    out1 = outr.logsoftmax()
    outm = out1.mul(m)
    outx = outm.sum()
    outx.backward()

    return outx.data, x.grad, W.grad


if __name__ == "__main__":

    x_init = np.random.randn(1, 3).astype(np.float32)
    W_init = np.random.randn(3, 3).astype(np.float32)
    m_init = np.random.randn(1, 3).astype(np.float32)

    print(test_tinygrad())
