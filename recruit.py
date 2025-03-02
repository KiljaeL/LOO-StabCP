import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

from funcs.CP_screening import cfBH, RO_cfBH, LOO_cfBH
from funcs.post import post_recruit, figure_3

np.random.seed(42)
warnings.filterwarnings("ignore")

T = 1000; R = 15
qgrid = np.array([0.1,0.2,0.3])

recruit = pd.read_csv('recruit.csv', index_col=0)
recruit = recruit.drop(columns=['salary'])
recruit = pd.get_dummies(recruit, columns=['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status'], drop_first=True)

X_recruit_full = recruit.drop(columns=['status_Placed']).values
mx = np.mean(X_recruit_full, axis=0); sx = np.std(X_recruit_full, axis=0)
X_recruit_full = (X_recruit_full - mx) / sx / np.sqrt(X_recruit_full.shape[1] + 1)
X_recruit_full = np.concatenate((np.ones((X_recruit_full.shape[0], 1))/np.sqrt(X_recruit_full.shape[1] + 1), X_recruit_full), axis=1)
Y_recruit_full = recruit['status_Placed'].values

saves = np.zeros((T, 3, len(qgrid), 3))

print('Starting Conformalized Screening...')
for t in range(T):
    if t % 100 == 0:
        print(f'Iteration {t} out of {T}')
        X_recruit_train, X_recruit_test, Y_recruit_train, Y_recruit_test = train_test_split(X_recruit_full, Y_recruit_full, test_size=0.3)
    D_recruit = (X_recruit_train, Y_recruit_train)
    D_recruit_test = (X_recruit_test, Y_recruit_test)
    shuffles = [np.random.permutation(X_recruit_train.shape[0] + 1) for _ in range(R)]
    for i, q in enumerate(qgrid):
        saves[t, 0, i, :] = cfBH(D_recruit, D_recruit_test, A = 'SGD', q=q, params_sgd={'shuffles': shuffles, 'thres': 1, 'lr': 0.001, 'epochs': R})
        saves[t, 1, i, :] = RO_cfBH(D_recruit, D_recruit_test, A = 'SGD', q=q, params_sgd={'shuffles': shuffles, 'thres': 1, 'lr': 0.001, 'epochs': R})
        saves[t, 2, i, :] = LOO_cfBH(D_recruit, D_recruit_test, A = 'SGD', q=q, params_sgd={'shuffles': shuffles, 'thres': 1, 'lr': 0.001, 'epochs': R})
        
results = post_recruit(saves, T, qgrid)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
sns.set_palette("Set2")
figure_3(results, qgrid)

print('Results from Application (Conformalized Screening):')
print(results.groupby(['q', 'Method']).agg({'Coverage': ['mean', 'std'], 'Length': ['mean', 'std'], 'Time': ['mean', 'std']}).round(3))