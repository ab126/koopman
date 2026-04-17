import numpy as np

def gauss_process(t, sigma=2.0):
    u = sigma * np.random.randn(len(t))
    return u



