import numpy as np
import time
from funcs.utils import robust_RLM, robust_SGD

def OracleCP(D, D_test, A, alpha=0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffles': None, 'lr': 0.01, 'epochs': 1}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    Coverages = np.zeros(m); Sizes = np.zeros(m)

    start = time.time()
    for i, x_test in enumerate(X_test):
        y_test = Y_test[i]
        X_aug = X.copy(); Y_aug = Y.copy()
        X_aug = np.vstack([X_aug, x_test]); Y_aug = np.append(Y_aug, y_test)
        if A == 'RLM':
            theta = robust_RLM((X_aug, Y_aug), thres=params_rlm['thres'], lamb=params_rlm['lamb'])
        elif A == 'SGD':
            theta = robust_SGD((X_aug, Y_aug), shuffles=params_sgd['shuffles'], lr=params_sgd['lr'], epochs=params_sgd['epochs'])
        Y_pred = X_aug @ theta
        S = np.abs(Y_pred - Y_aug)
        Q = np.quantile(S, 1-alpha, interpolation='higher')
        Coverages[i] = np.abs(Y_pred[n] - y_test) <= Q
        Sizes[i] = 2 * Q
        
    end = time.time()
    Time = end - start
    Coverage = Coverages.mean()
    Size = Sizes.mean()
    return Coverage, Size, Time

def FullCP(D, D_test, A, alpha=0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffles': None, 'lr': 0.01, 'epochs': 1}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    Z = np.linspace(Y.min() - Y.std(), Y.max() + Y.std(), 100)
    Coverages = np.zeros(m); Sizes = np.zeros(m)

    start = time.time()
    for i, x_test in enumerate(X_test):
        y_test = Y_test[i]
        FCP = [-np.inf, np.inf]; pointer = 0
        while FCP[0] == -np.inf or FCP[1] == np.inf:
            z = Z[pointer]
            X_aug = X.copy(); Y_aug = Y.copy()
            X_aug = np.vstack([X_aug, x_test]); Y_aug = np.append(Y_aug, z)
            
            if A == 'RLM':
                theta = robust_RLM((X_aug, Y_aug), thres=params_rlm['thres'], lamb=params_rlm['lamb'])
            elif A == 'SGD':
                theta = robust_SGD((X_aug, Y_aug), shuffles=params_sgd['shuffles'], lr=params_sgd['lr'], epochs=params_sgd['epochs'])
            Y_pred = X_aug @ theta
            S = np.abs(Y_pred - Y_aug)
            Q = np.quantile(S, 1-alpha, interpolation='higher')
            
            if FCP[0] == -np.inf:
                if S[n] <= Q:
                    FCP[0] = z
                    pointer = len(Z) - 1
                    continue
                else:
                    pointer += 1
                    continue

            if FCP[1] == np.inf:
                if S[n] <= Q:
                    FCP[1] = z
                    continue
                else:
                    pointer -= 1
                    continue
        Coverages[i] = (y_test >= FCP[0]) & (y_test <= FCP[1])
        Sizes[i] = FCP[1] - FCP[0]
    
    end = time.time()
    Time = end - start
    Coverage = Coverages.mean()
    Size = Sizes.mean()
    return Coverage, Size, Time

def SplitCP(D, D_test, A, alpha=0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffles': None, 'lr': 0.01, 'epochs': 1}):
    X, Y = D[0], D[1]; n = len(Y)
    
    n_train = int(n * 0.7)
    ind_train = np.random.choice(n, n_train, replace=False)
    ind_calib = np.setdiff1d(np.arange(n), ind_train)
    X_train, Y_train = X[ind_train], Y[ind_train]
    X_calib, Y_calib = X[ind_calib], Y[ind_calib]
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    Coverages = np.zeros(m)

    start = time.time()
    if A == 'RLM':
        theta = robust_RLM((X_train, Y_train), thres=params_rlm['thres'], lamb=params_rlm['lamb'])
    elif A == 'SGD':
        theta = robust_SGD((X_train, Y_train), shuffles=params_sgd['shuffles'], lr=params_sgd['lr'], epochs=params_sgd['epochs'])
    Y_calib_pred = X_calib @ theta
    S = np.abs(Y_calib_pred - Y_calib)
    Q = np.quantile(S, 1-alpha, interpolation='higher')
    
    Y_test_pred = X_test @ theta
    for i, y_test in enumerate(Y_test):
        Coverages[i] = np.abs(Y_test_pred[i] - y_test) <= Q
    end = time.time()

    Time = end - start
    Coverage = Coverages.mean()
    Size = 2 * Q
    return Coverage, Size, Time

def RO_StabCP(D, D_test, A, alpha=0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffles': None, 'lr': 0.01, 'epochs': 1}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    zhat = 0
    if A == "RLM" or A == "SGD":
        nu = np.linalg.norm(X, axis=1)
        eps = params_rlm['thres']
        if A == "RLM":
            lamb = params_rlm['lamb']
        if A == "SGD":
            eta = params_sgd['lr']
    Coverages = np.zeros(m); Sizes = np.zeros(m)

    start = time.time()
    for i, x_test in enumerate(X_test):
        y_test = Y_test[i]
        X_aug = X.copy(); Y_aug = Y.copy()
        X_aug = np.vstack([X_aug, x_test]); Y_aug = np.append(Y_aug, zhat)
        if A == "RLM" or A == "SGD":
            nu_new = np.linalg.norm(x_test)
            nu_aug = np.append(nu, nu_new)
            rho_new = eps*nu_new.copy()
            if A == "RLM":
                tau_aug = 4*(nu_aug*rho_new)/(n+1)/lamb
                theta = robust_RLM((X_aug, Y_aug), thres=params_rlm['thres'], lamb=params_rlm['lamb'])
            elif A == "SGD":
                R = params_sgd['epochs']
                tau_aug = 2*R*eta*rho_new*nu_aug
                theta = robust_SGD((X_aug, Y_aug), shuffles=params_sgd['shuffles'], lr=params_sgd['lr'], epochs=params_sgd['epochs'])
            
        Y_pred = X_aug @ theta
        S = np.abs(Y_pred[:n] - Y_aug[:n])
        U = S + tau_aug[:n]
        Q = np.quantile(U, 1-alpha, interpolation='higher') + tau_aug[n]
        
        Coverages[i] = np.abs(Y_pred[n] - y_test) <= Q
        Sizes[i] = 2 * Q
    
    end = time.time()
    Time = end - start
    Coverage = Coverages.mean()
    Size = Sizes.mean()
    return Coverage, Size, Time

def LOO_StabCP(D, D_test, A, alpha=0.1, params_rlm = {'thres': 1, 'lamb': 1}, params_sgd = {'shuffles': None, 'lr': 0.01, 'epochs': 1}):
    X, Y = D[0], D[1]; n = len(Y)
    X_test, Y_test = D_test[0], D_test[1]; m = len(Y_test)
    if A == "RLM" or A == "SGD":
        nu = np.linalg.norm(X, axis=1)
        eps = params_rlm['thres']
        if A == "RLM":
            lamb = params_rlm['lamb']
            rho = eps*nu.copy()
            rhobar = np.mean(rho)
        elif A == "SGD":
            eta = params_sgd['lr']
    Coverages = np.zeros(m); Sizes = np.zeros(m)

    start = time.time()
    if A == "RLM":
        theta = robust_RLM((X, Y), thres=params_rlm['thres'], lamb=params_rlm['lamb'])
    elif A == "SGD":
        R = params_sgd['epochs']
        theta = robust_SGD((X, Y), shuffles=params_sgd['shuffles'], lr=params_sgd['lr'], epochs=params_sgd['epochs'])
    Y_pred = X @ theta
    S = np.abs(Y_pred - Y)
    
    for i, x_test in enumerate(X_test):
        y_test = Y_test[i]
        if A == "RLM" or A == "SGD":
            nu_new = np.linalg.norm(x_test)
            nu_aug = np.append(nu, nu_new)
            rho_new = eps*nu_new.copy()
            if A == "RLM":
                tau_aug = 2*nu_aug*(rho_new + rhobar)/(n+1)/lamb
            elif A == "SGD":
                tau_aug = R*eta*rho_new*nu_aug
         
        U = S + tau_aug[:n]
        Q = np.quantile(U, 1-alpha, interpolation='higher') + tau_aug[n]

        y_test_pred = x_test.reshape(1, -1) @ theta
        y_test_pred = y_test_pred[0]
        Coverages[i] = np.abs(y_test_pred - y_test) <= Q
        Sizes[i] = 2 * Q
    
    end = time.time()
    Time = end - start
    Coverage = Coverages.mean()
    Size = Sizes.mean()
    return Coverage, Size, Time