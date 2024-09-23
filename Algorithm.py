import numpy as np
import cvxpy as cp

def Robust_RLM(D, thres = 1, lamb = 1):
    X, Y = D
    n, p = X.shape
    
    theta = cp.Variable(p)
    lmd = cp.Parameter(nonneg=True)
    lmd.value = lamb

    def objective(X, Y, theta, thres, lmd):
        return cp.mean(cp.huber(Y - X @ theta, thres))/2 + lamb/2 * cp.norm2(theta)**2

    problem = cp.Problem(cp.Minimize(objective(X, Y, theta, thres, lmd)))
    problem.solve()

    return theta.value

def Robust_SGD(D, shuffle, thres=1, lr=0.001, epochs=1):
    X, Y = D
    n, p = X.shape
    theta = np.zeros(p)
    for epoch in range(epochs):
        for i in shuffle:
            if i >= n:
                continue
            x = X[i]
            y = Y[i]
            prediction = x @ theta
            residual = y - prediction
            if abs(residual) <= thres:
                gradient = residual * x
            else:
                gradient = thres * np.sign(residual) * x
            theta += lr * gradient
    return theta