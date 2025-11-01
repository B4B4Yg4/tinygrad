import requests
import os
import gzip
import hashlib
import numpy as np


def fetch(url):

    fp = os.path.join("/tmp", hashlib.md5(url.encode("utf-8")).hexdigest())

    if os.path.isfile(fp):

        with open(fp, "rb") as f:
            data = f.read()

    else:

        with open(fp, "wb") as f:

            data = requests.get(url).content
            f.write(data)

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def fetch_mnist():

    Y_train = fetch(
        "https://systemds.apache.org/assets/datasets/mnist/train-labels-idx1-ubyte.gz"
    )[8:]
    X_train = fetch(
        "https://systemds.apache.org/assets/datasets/mnist/train-images-idx3-ubyte.gz"
    )[0x10:].reshape((-1, 28, 28))
    X_test = fetch(
        "https://systemds.apache.org/assets/datasets/mnist/t10k-images-idx3-ubyte.gz"
    )[0x10:].reshape((-1, 28, 28))
    Y_test = fetch(
        "https://systemds.apache.org/assets/datasets/mnist/t10k-labels-idx1-ubyte.gz"
    )[8:]

    return X_train, X_test, Y_train, Y_test
