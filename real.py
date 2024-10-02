import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

from funcs.CP import OracleCP, FullCP, SplitCP, RO_StabCP, LOO_StabCP
from funcs.utils import get_data
from funcs.post import post_real, figure_2

np.random.seed(42)
warnings.filterwarnings("ignore")

T = 100
mgrid = [1, 100]
R_full = 5
R = 15

X_boston_full, Y_boston_full = get_data('boston')
X_diabetes_full, Y_diabetes_full = get_data('diabetes')

save_Boston_RLM = np.zeros((T, 5, len(mgrid), 3))
save_Boston_SGD = np.zeros((T, 5, len(mgrid), 3))
save_Diabetes_RLM = np.zeros((T, 5, len(mgrid), 3))
save_Diabetes_SGD = np.zeros((T, 5, len(mgrid), 3))

for i, m in enumerate(mgrid):
    print(f'Starting m = {m}...')
    print('Starting Boston...')
    for t in range(T):
        if (t + 1) % 10 == 0 or t < 10:
            print(f'Iteration {t+1} out of {T}')
        X_boston, X_boston_test, Y_boston, Y_boston_test = train_test_split(X_boston_full, Y_boston_full, test_size=int(m))
        D_boston, D_boston_test = (X_boston, Y_boston), (X_boston_test, Y_boston_test)
        params_rlm = {'thres': 1, 'lamb': 2}
        shuffles_full = [np.random.permutation(len(Y_boston)+1) for _ in range(R_full)]
        params_sgd_full = {'shuffles': shuffles_full, 'lr': 0.001, 'epochs': R_full}
        shuffles = [np.random.permutation(len(Y_boston)+1) for _ in range(R)]
        params_sgd = {'shuffles': shuffles, 'lr': 0.001, 'epochs': R}
        
        save_Boston_RLM[t, 0, i, :] = OracleCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 1, i, :] = FullCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 2, i, :] = SplitCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 3, i, :] = RO_StabCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 4, i, :] = LOO_StabCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        
        save_Boston_SGD[t, 0, i, :] = OracleCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
        save_Boston_SGD[t, 1, i, :] = FullCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd_full)
        save_Boston_SGD[t, 2, i, :] = SplitCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
        save_Boston_SGD[t, 3, i, :] = RO_StabCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
        save_Boston_SGD[t, 4, i, :] = LOO_StabCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
    
    print('Starting Diabetes...')
    for t in range(T):
        if (t + 1) % 10 == 0 or t < 10:
            print(f'Iteration {t+1} out of {T}')
        X_diabetes, X_diabetes_test, Y_diabetes, Y_diabetes_test = train_test_split(X_diabetes_full, Y_diabetes_full, test_size=int(m))
        D_diabetes, D_diabetes_test = (X_diabetes, Y_diabetes), (X_diabetes_test, Y_diabetes_test)
        
        params_rlm = {'thres': 1, 'lamb': 2}
        shuffles_full = [np.random.permutation(len(Y_diabetes)+1) for _ in range(R_full)]
        params_sgd_full = {'shuffles': shuffles_full, 'lr': 0.001, 'epochs': R_full}
        shuffles = [np.random.permutation(len(Y_diabetes)+1) for _ in range(R)]
        params_sgd = {'shuffles': shuffles, 'lr': 0.001, 'epochs': R}
        
        
        save_Diabetes_RLM[t, 0, i, :] = OracleCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 1, i, :] = FullCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 2, i, :] = SplitCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 3, i, :] = RO_StabCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 4, i, :] = LOO_StabCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        
        save_Diabetes_SGD[t, 0, i, :] = OracleCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 1, i, :] = FullCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd_full)
        save_Diabetes_SGD[t, 2, i, :] = SplitCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 3, i, :] = RO_StabCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 4, i, :] = LOO_StabCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        
saves = [save_Boston_RLM, save_Boston_SGD, save_Diabetes_RLM, save_Diabetes_SGD]
results = post_real(saves, T, mgrid)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 11.3
sns.set_palette("Set2")
figure_2(results, mgrid)

print('Results from Real Data Examples:')
print(results.groupby(['m', 'Dataset', 'Algorithm','Method']).agg({'Coverage': ['mean', 'std'], 'Length': ['mean', 'std'], 'Time': ['mean', 'std']}).round(3))
