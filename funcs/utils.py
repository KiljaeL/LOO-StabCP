import numpy as np
import cvxpy as cp
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml

def robust_RLM(D, thres = 1, lamb = 1):
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

def robust_SGD(D, shuffles, thres=1, lr=0.001, epochs=1):
    X, Y = D
    n, p = X.shape
    theta = np.zeros(p)
    for epoch in range(epochs):
        for i in shuffles[epoch]:
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

def BH(pval, q = 0.1):
    ntest = len(pval)
    
    df_test = pd.DataFrame({"id": range(ntest), "pval": pval}).sort_values(by='pval')
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)

def DGP_lin(n,m,p,sigma=1):
    X = np.random.randn(n,p)/np.sqrt(p)
    X_test = np.random.randn(m,p)/np.sqrt(p)
    theta = np.array([1 - i/p for i in range(1,p+1)])**5
    theta = theta/np.linalg.norm(theta)/np.sqrt(p)
    Y = X @ theta + np.random.randn(n) * sigma
    Y_test = X_test @ theta + np.random.randn(m) * sigma
    D = (X,Y)
    D_test = (X_test,Y_test)
    return D,D_test

def DGP_nonlin(n,m,p,sigma=1):
    X = np.random.randn(n,p)/np.sqrt(p)
    X_test = np.random.randn(m,p)/np.sqrt(p)
    theta = np.array([1 - i/p for i in range(1,p+1)])**5
    theta = theta/np.linalg.norm(theta)/np.sqrt(p)
    Y = (np.exp(X/10)) @ theta + np.random.randn(n) * sigma
    Y_test = (np.exp(X_test/10)) @ theta + np.random.randn(m) * sigma
    D = (X,Y)
    D_test = (X_test,Y_test)
    return D,D_test
    
def get_data(dataname):
    if dataname == 'boston':
        dataset = fetch_openml(name='boston', version=1)
    elif dataname == 'diabetes':
        dataset = load_diabetes()
    X = np.array(dataset.data, dtype=float); Y = np.array(dataset.target)
    mx = np.mean(X, axis=0); my = np.mean(Y)
    sx = np.std(X, axis=0); sy = np.std(Y)
    X = (X - mx) / sx / np.sqrt(X.shape[1]); Y = (Y - my) / sy
    return X, Y
