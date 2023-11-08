from sympy import *
import numpy as np

betas = np.random.random(128)
mus = np.random.random(128)
pis = np.random.random(128)

def cdf(x, mus, betas, pis):
    return np.sum(pis * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas))))


def pdf(x, mus, betas, pis):
    return np.sum(pis * (1 / (2 * betas)) * np.exp(-np.abs(x - mus) / betas))


x = Symbol('x')
y = cdf(x, mus, betas, pis)
yprime = y.diff(x)
print(y_prime)