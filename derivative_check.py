from autodiff import function, gradient
import numpy as np

betas = np.random.random(128)
mus = np.random.random(128)
pis = np.random.random(128)

@gradient
def cdf(x, mus, betas, pis):
    return np.sum(pis * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas))))


def pdf(x, mus, betas, pis):
    return np.sum(pis * (1 / (2 * betas)) * np.exp(-np.abs(x - mus) / betas))


x = 0
y_autodiff = cdf(x, mus, betas, pis)
y = pdf(x, mus, betas, pis)
print(y_autodiff)
print(y)