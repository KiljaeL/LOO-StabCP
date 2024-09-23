import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from DGP import DGP_lin, DGP_nonlin
from CP import OracleCP, FullCP, SplitCP, RO_StabCP, LOO_StabCP
from Postprocessing import Post_synthetic, Figure_1

np.random.seed(42)
warnings.filterwarnings("ignore")

T = 100
n, m, d = 100, 100, 100

save_lin_RLM = np.zeros((T, 5, 3))
save_lin_SGD = np.zeros((T, 5, 3))
save_nonlin_RLM = np.zeros((T, 5, 3))
save_nonlin_SGD = np.zeros((T, 5, 3))

print('Starting Linear DGP...')
for t in range(T):
    if (t + 1) % 10 == 0 or t < 10:
        print(f'Iteration {t+1} out of {T}')
    D, D_test = DGP_lin(n, m, d)
    params_rlm = {'thres': 1, 'lamb': 2}
    params_sgd = {'shuffle': np.random.permutation(n + 1), 'lr': 0.01}

    save_lin_RLM[t, 0, :] = OracleCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_lin_RLM[t, 1, :] = FullCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_lin_RLM[t, 2, :] = SplitCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_lin_RLM[t, 3, :] = RO_StabCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_lin_RLM[t, 4, :] = LOO_StabCP(D, D_test,'RLM', params_rlm = params_rlm)
    
    save_lin_SGD[t, 0, :] = OracleCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_lin_SGD[t, 1, :] = FullCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_lin_SGD[t, 2, :] = SplitCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_lin_SGD[t, 3, :] = RO_StabCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_lin_SGD[t, 4, :] = LOO_StabCP(D, D_test, 'SGD', params_sgd = params_sgd)

print('Starting Noninear DGP...')
for t in range(T):
    if (t + 1) % 10 == 0 or t < 10:
        print(f'Iteration {t+1} out of {T}')
    D, D_test = DGP_nonlin(n, m, d)
    params_rlm = {'thres': 1, 'lamb': 2}
    params_sgd = {'shuffle': np.random.permutation(n + 1), 'lr': 0.01}

    save_nonlin_RLM[t, 0, :] = OracleCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_nonlin_RLM[t, 1, :] = FullCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_nonlin_RLM[t, 2, :] = SplitCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_nonlin_RLM[t, 3, :] = RO_StabCP(D, D_test, 'RLM', params_rlm = params_rlm)
    save_nonlin_RLM[t, 4, :] = LOO_StabCP(D, D_test,'RLM', params_rlm = params_rlm)
    
    save_nonlin_SGD[t, 0, :] = OracleCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_nonlin_SGD[t, 1, :] = FullCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_nonlin_SGD[t, 2, :] = SplitCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_nonlin_SGD[t, 3, :] = RO_StabCP(D, D_test, 'SGD', params_sgd = params_sgd)
    save_nonlin_SGD[t, 4, :] = LOO_StabCP(D, D_test, 'SGD', params_sgd = params_sgd)

saves = [save_lin_RLM, save_lin_SGD, save_nonlin_RLM, save_nonlin_SGD]
results = Post_synthetic(saves, T)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 9
sns.set_palette("Set2")
Figure_1(results)

print('Results for Synthetic Data (Mean):')
print(results.groupby(['DGP', 'Algorithm','Method']).mean())
print('\nResults for Synthetic Data (SD):')
print(results.groupby(['DGP', 'Algorithm','Method']).std())