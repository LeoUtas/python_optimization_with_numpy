import numpy as np


def make_dummy_y_unvectorized1(x, vector_w, b, error_term):
    y = []
    m = x.shape[1]
    for i in range(m):
        y_i = 0
        for j in range(len(vector_w)):
            y_i += vector_w[j] * x[j, i]
        y_i = (y_i + b) * np.exp(error_term[i])

        y.append(y_i)
        y = np.array(y)
    return y


def make_dummy_y_unvectorized2(x, vector_w, b, error_term):
    m, n = x.shape
    y = np.zeros(n)
    for i in range(n):
        for j in range(m):
            y[i] += vector_w[j] * x[j, i]
    y = (y + b) * np.exp(error_term)
    return y


def make_dummy_y_vectorized1(x, vector_w, b, error_term):
    y = []
    for i in range(x.shape[1]):
        y.append((np.dot(vector_w, x[:, i]) + b) * np.exp(error_term[i]))
        y = np.array(y)
    return y


def make_dummy_y_vectorized2(x, vector_w, b, error_term):
    y = (np.dot(vector_w, x) + b) * np.exp(error_term)
    return y


def measure_execution_time(func, *args, **kwargs):
    from time import time

    start_time = time()
    func(*args, **kwargs)
    end_time = time()
    estimate = round(end_time - start_time, 4)

    return estimate
