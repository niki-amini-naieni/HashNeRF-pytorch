import numpy as np
from findiff import FinDiff

betas = np.random.random(128)
mus = np.random.random(128)
pis = np.random.random(128)

def cdf(x, mus, betas, pis):
    x = np.stack((x,)*128, axis=-1)
    return np.sum(pis * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas))), axis=-1)


def pdf(x, mus, betas, pis):
    x = np.stack((x,)*128, axis=-1)
    return np.sum(pis * (1 / (2 * betas)) * np.exp(-np.abs(x - mus) / betas), axis=-1)


x = np.linspace(0, 1, 200)
y = cdf(x, mus, betas, pis)
print(x.shape)
print(y.shape)
d_dx = FinDiff(0, x[1] - x[0], 1, acc=2)
dy_dx = d_dx(y)
print(np.max(abs(dy_dx - pdf(x, mus, betas, pis))))
