import numpy as np
from pyDOE import lhs

def u_true(x):
    X = np.sum(x*x, axis=1, keepdims=True)
    return np.exp(-10*X)

def generate_peak1_samples(N_b, N_f, lb, ub, Dimension=10):
    # np.random.seed(1)
    X_f = lb + (ub-lb)*lhs(Dimension, N_f)
    # X_f = np.random.uniform(lb[0], ub[0], (N_f, Dimension))

    X_b_train = [] 
    n = N_b // 20
    for i in range(Dimension):
        X_b_new = np.random.uniform(lb[0], ub[0], (2*n, Dimension))
        X_b_new[:n, i] = lb[i]
        X_b_new[n:, i] = ub[i]
        X_b_train.append(X_b_new)
    X_b_train = np.vstack(X_b_train)
    u_b = u_true(X_b_train)
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_b_train = X_b_train[index]
    u_b = u_b[index]
    return X_f, X_b_train, u_b
    