import numpy as np

def DGP_lin(n,m,p,sigma=1):
    X = np.random.randn(n,p)/np.sqrt(p)
    X_test = np.random.randn(m,p)/np.sqrt(p)
    theta = np.array([1 - i/p for i in range(1,p+1)])**5
    theta = theta/np.linalg.norm(theta)*np.sqrt(1/p)
    Y = X @ theta + np.random.randn(n) * sigma
    Y_test = X_test @ theta + np.random.randn(m) * sigma
    D = (X,Y)
    D_test = (X_test,Y_test)
    return D,D_test

def DGP_nonlin(n,m,p,sigma=1):
    X = np.random.randn(n,p)/np.sqrt(p)
    X_test = np.random.randn(m,p)/np.sqrt(p)
    theta = np.array([1 - i/p for i in range(1,p+1)])**5
    theta = theta/np.linalg.norm(theta)*np.sqrt(1/p)
    Y = (np.exp(X/10)) @ theta + np.random.randn(n) * sigma
    Y_test = (np.exp(X_test/10)) @ theta + np.random.randn(m) * sigma
    D = (X,Y)
    D_test = (X_test,Y_test)
    return D,D_test