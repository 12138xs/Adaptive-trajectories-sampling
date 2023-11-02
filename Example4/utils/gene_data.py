import numpy as np
import random
from pyDOE import lhs

def u_true(x):
    X = np.mean(x, axis=1).reshape(-1, 1)
    return X*X + np.sin(X)

def generate_peak1_samples(N_b, N_f, lb, ub, Dimension = 10):
    # np.random.seed(1)
    X_f = lb + (ub-lb)*lhs(Dimension, N_f)
    # X_f = np.random.uniform(lb[0], ub[0], (N_f, Dimension))

    # X_b_train = [] 
    # n = N_b // 20
    # for i in range(Dimension):
    #     X_b_new = np.random.uniform(lb[0], ub[0], (2*n, Dimension))
    #     X_b_new[:n, i] = lb[i]
    #     X_b_new[n:, i] = ub[i]
    #     X_b_train.append(X_b_new)
    # X_b_train = np.vstack(X_b_train)
    x = np.random.uniform(low=-1, high=1, size=(N_f, Dimension))
    for i in range(N_b):
        idx_num1 = random.randint(0, Dimension - 1)
        if idx_num1 == 0:
            idx_num2 = random.randint(1, Dimension - 1)
        else:
            idx_num2 = random.randint(0, Dimension - 1)

        idx1 = random.sample(list(range(Dimension)), idx_num1)
        x[i, idx1] = 1

        idx2 = random.sample(list(range(Dimension)), idx_num2)
        x[i, idx2] = -1
    X_b_train = x
    u_b = u_true(X_b_train)
    index = np.arange(0, N_b)
    np.random.shuffle(index)
    X_b_train = X_b_train[index]
    u_b = u_b[index]
    return X_f, X_b_train, u_b
    
