import numpy as np
import time

from Algorithm import Robust_RLM, Robust_SGD
from BH import BH


def cfBH(D, D_test, A, q = 0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffle': None, 'thres': 1, 'lr': 0.01}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    
    n_train = int(0.5*n)
    n_calib = n - n_train
    ind_train = np.random.choice(n, n_train, replace=False)
    ind_calib = np.setdiff1d(range(n), ind_train)
    X_train, X_calib, Y_train, Y_calib = X[ind_train], X[ind_calib], Y[ind_train], Y[ind_calib]
    D_train = (X_train, 1*(Y_train>0))
    
    start = time.time()
    if A == 'RLM':
        thres, lamb = params_rlm['thres'], params_rlm['lamb']
        theta = Robust_RLM(D_train, thres, lamb)
    elif A == 'SGD':
        shuffle, thres, lr = params_sgd['shuffle'], params_sgd['thres'], params_sgd['lr']
        theta = Robust_SGD(D_train, shuffle, thres, lr)

    V = 1000 * (Y_calib > 0) - X_calib @ theta
    Vhat = - X_test @ theta
    
    pval = np.array([np.sum(V < Vhat[j]) + 1 for j in range(m)])/(n_calib+1)
    pval = pval.flatten()

    selected = BH(pval,q)
    end = time.time()
    
    if len(selected) == 0:
        FDP = 0
        Power = 0
        Time = end-start
    else:
        FDP = np.sum(Y_test[selected] <= 0)/len(selected)
        Power = np.sum(Y_test[selected] > 0) / sum(Y_test > 0)
        Time = end-start
        
    return FDP, Power, Time

def RO_cfBH(D, D_test, A, q = 0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffle': None, 'thres': 1, 'lr': 0.01}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    zhat = 0
    
    start = time.time()
    nu = np.linalg.norm(X, axis=1)
    eps = params_rlm['thres']
    if A == 'RLM':
        thres, lamb = params_rlm['thres'], params_rlm['lamb']
    elif A == 'SGD':
        shuffle, thres, lr = params_sgd['shuffle'], params_sgd['thres'], params_sgd['lr']
    
    pval = np.zeros(m)
    for j, x_test in enumerate(X_test):
        X_aug = X.copy(); Y_aug = Y.copy()
        X_aug = np.vstack((X, x_test)); Y_aug = np.append(1*(Y>0), zhat)        
        nu_new = np.linalg.norm(x_test)
        nu_aug = np.append(nu, nu_new)
        rho_new = eps*nu_new.copy()
        if A == 'RLM':
            tau_aug = 4*(nu_aug*rho_new)/(n+1)/lamb
            theta = Robust_RLM((X_aug, Y_aug), thres, lamb)
        if A == 'SGD':
            tau_aug = 2*lr*rho_new*nu_aug
            theta = Robust_SGD((X_aug, Y_aug), shuffle=shuffle, lr=lr)
            
        V = 1000 * (Y > 0) - X @ theta
        Vhat = 1000 * zhat - x_test @ theta
        pval[j] = (np.sum((V - tau_aug[:n] < Vhat + tau_aug[-1])) + 1)/(n+1)
        
    selected = BH(pval,q)
    end = time.time()

    if len(selected) == 0:
        FDP = 0
        Power = 0
        Time = end-start
    else:
        FDP = np.sum(Y_test[selected] <= 0)/len(selected)
        Power = np.sum(Y_test[selected] > 0) / sum(Y_test > 0)
        Time = end-start
        
    return FDP, Power, Time
    
def LOO_cfBH(D, D_test, A, q = 0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffle': None, 'thres': 1, 'lr': 0.01}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    D_train = (X, 1*(Y>0))
    
    start = time.time()
    nu = np.linalg.norm(X, axis=1)
    eps = params_rlm['thres']
    if A == 'RLM':
        thres, lamb = params_rlm['thres'], params_rlm['lamb']
        rho = eps*nu.copy()
        rhobar = np.mean(rho)
        theta = Robust_RLM(D_train, thres, lamb)
    elif A == 'SGD':
        shuffle, thres, lr = params_sgd['shuffle'], params_sgd['thres'], params_sgd['lr']
        theta = Robust_SGD(D_train, shuffle, thres, lr)
        
    V = 1000 * (Y > 0) - X @ theta
    Vhat = - X_test @ theta # M * (0 > 0).astype(int) = 0

    nu_test = np.linalg.norm(X_test, axis=1)
    rho_test = eps*nu_test.copy()

    pval = np.zeros(m)
    if A == 'RLM':
        for j in range(m):
            tau = 2*nu*(rho_test[j] + rhobar)/(n+1)/lamb
            tau_test = 2*nu_test[j]*(rho_test[j] + rhobar)/(n+1)/lamb
            pval[j] = (np.sum((V - tau < Vhat[j] + tau_test)) + 1)/(n+1)
    elif A == 'SGD':
        for j in range(m):
            tau = lr*nu*rho_test[j]
            tau_test = lr*nu_test[j]*rho_test[j]
            pval[j] = (np.sum((V - tau < Vhat[j] + tau_test)) + 1)/(n+1)

    selected = BH(pval,q)
    end = time.time()
    
    if len(selected) == 0:
        FDP = 0
        Power = 0
        Time = end-start
    else:
        FDP = np.sum(Y_test[selected] <= 0)/len(selected)
        Power = np.sum(Y_test[selected] > 0) / sum(Y_test > 0)
        Time = end-start
        
    return FDP, Power, Time