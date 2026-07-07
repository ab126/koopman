import numpy as np

def gauss_process(t, sigma=1.0):
    u = sigma * np.random.randn(len(t))
    return u

# Manifold control Utils



