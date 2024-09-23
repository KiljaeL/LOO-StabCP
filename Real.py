import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
import warnings

from CP import OracleCP, FullCP, SplitCP, RO_StabCP, LOO_StabCP
from Postprocessing import Post_real, Figure_2

np.random.seed(42)
warnings.filterwarnings("ignore")

T = 100
mgrid = [1, 10, 100]

boston = fetch_openml(name='boston', version=1)
X_boston_full = np.array(boston.data, dtype=float)
Y_boston_full = np.array(boston.target)
mx = np.mean(X_boston_full, axis=0); my = np.mean(Y_boston_full)
sx = np.std(X_boston_full, axis=0); sy = np.std(Y_boston_full)
X_boston_full = (X_boston_full - mx) / sx / np.sqrt(X_boston_full.shape[1])
Y_boston_full = (Y_boston_full - my) / sy

diabetes = load_diabetes()
X_diabetes_full = diabetes.data
Y_diabetes_full = diabetes.target
mx = np.mean(X_diabetes_full, axis=0); my = np.mean(Y_diabetes_full)
sx = np.std(X_diabetes_full, axis=0); sy = np.std(Y_diabetes_full)
X_diabetes_full = (X_diabetes_full - mx) / sx / np.sqrt(X_diabetes_full.shape[1])
Y_diabetes_full = (Y_diabetes_full - my) / sy

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
        shuffle = np.random.permutation(len(Y_boston)+1)
        params_sgd = {'shuffle': shuffle, 'lr': 0.001}
        
        save_Boston_RLM[t, 0, i, :] = OracleCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 1, i, :] = FullCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 2, i, :] = SplitCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 3, i, :] = RO_StabCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        save_Boston_RLM[t, 4, i, :] = LOO_StabCP(D_boston, D_boston_test, 'RLM', params_rlm = params_rlm)
        
        save_Boston_SGD[t, 0, i, :] = OracleCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
        save_Boston_SGD[t, 1, i, :] = FullCP(D_boston, D_boston_test, 'SGD', params_sgd = params_sgd)
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
        shuffle = np.random.permutation(len(Y_diabetes)+1)
        params_sgd = {'shuffle': shuffle, 'lr': 0.001}
        
        save_Diabetes_RLM[t, 0, i, :] = OracleCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 1, i, :] = FullCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 2, i, :] = SplitCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 3, i, :] = RO_StabCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        save_Diabetes_RLM[t, 4, i, :] = LOO_StabCP(D_diabetes, D_diabetes_test, 'RLM', params_rlm = params_rlm)
        
        save_Diabetes_SGD[t, 0, i, :] = OracleCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 1, i, :] = FullCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 2, i, :] = SplitCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 3, i, :] = RO_StabCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        save_Diabetes_SGD[t, 4, i, :] = LOO_StabCP(D_diabetes, D_diabetes_test, 'SGD', params_sgd = params_sgd)
        
saves = [save_Boston_RLM, save_Boston_SGD, save_Diabetes_RLM, save_Diabetes_SGD]
results = Post_real(saves, T, mgrid)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 9
sns.set_palette("Set2")
for variable in ['Coverage', 'Length', 'Time']:
    Figure_2(results, mgrid, variable)

print('Results for Real Data (Mean):')
print(results.groupby(['m', 'Dataset', 'Algorithm','Method']).mean())
print('\nResults for Real Data (SD):')
print(results.groupby(['m','Dataset', 'Algorithm','Method']).std())