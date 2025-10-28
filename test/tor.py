#!/usr/bin/env python

import warnings
import torch
import torch.nn as nn
import requests
import os
import gzip
import hashlib
import numpy as np
from tqdm import trange
warnings.filterwarnings("ignore")


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


class Model(nn.Module):

    def __init__(self) -> None:

        super(Model, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)

        return x


model = Model()

batch_size = 32
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

for _ in (t := trange(100)):

    samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
    X = torch.tensor(X_train[samp].reshape((-1, 28 * 28))).float()
    Y = torch.tensor(Y_train[samp])
    optim.zero_grad()
    out = model(X)
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == Y).float().mean()
    loss = loss_function(out, Y)
    loss.backward()
    optim.step()
    t.set_description(f"loss: {loss.item():.2f} accuracy: {accuracy.item():.2f}")


y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape(-1, 28 * 28)).float()), dim=1).numpy()
print((Y_test == y_test_preds).mean())
