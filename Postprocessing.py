import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Post_synthetic(saves, T):
    results = pd.concat([pd.DataFrame(saves[0].reshape(-1, 3)), \
                        pd.DataFrame(saves[1].reshape(-1, 3)), \
                        pd.DataFrame(saves[2].reshape(-1, 3)), \
                        pd.DataFrame(saves[3].reshape(-1, 3))], axis = 0)

    results.columns = ['Coverage', 'Length', 'Time']
    results['DGP'] = np.repeat(['Linear', 'Nonlinear'], T * 5 * 2)
    results['Method'] = np.tile(['OracleCP', 'FullCP', 'SplitCP', 'RO-StabCP', 'LOO-StabCP'], T * 2 * 2)
    results['Algorithm'] = np.tile(np.repeat(['RLM', 'SGD'], T * 5), 2)
    
    time_means = results.pivot_table(values='Time', index=['DGP', 'Algorithm'], columns='Method', aggfunc='mean').reset_index()
    timem = {
        (row['DGP'], row['Algorithm']): row['OracleCP']
        for _, row in time_means.iterrows()
    }
    results['Time'] = results.apply(lambda row: row['Time'] / timem[(row['DGP'], row['Algorithm'])], axis=1)
    results.to_csv('Result/results_synthetic.csv', index = False)
    return results

def Post_real(saves, T, mgrid):
    results = pd.concat([pd.DataFrame(saves[0].reshape(-1, 3)), \
                        pd.DataFrame(saves[1].reshape(-1, 3)), \
                        pd.DataFrame(saves[2].reshape(-1, 3)), \
                        pd.DataFrame(saves[3].reshape(-1, 3))], axis = 0)
    results.columns = ['Coverage', 'Length', 'Time']
    results['Dataset'] = np.repeat(['Boston', 'Diabetes'], T * 5 * len(mgrid) * 2)
    results['Method'] = np.tile(np.repeat(['OracleCP', 'FullCP', 'SplitCP', 'RO-StabCP', 'LOO-StabCP'], len(mgrid)), T * 2 * 2)
    results['Algorithm'] = np.tile(np.repeat(['RLM', 'SGD'], T * 5 * len(mgrid)), 2)
    results['m'] = np.tile(mgrid, T * 5 * 2 * 2).astype(str)
    
    time_means = results.pivot_table(values='Time', index=['Dataset', 'Algorithm', 'm'], columns='Method', aggfunc='mean').reset_index()
    timem = {
        (row['Dataset'], row['Algorithm'], row['m']): row['OracleCP']
        for _, row in time_means.iterrows()
    }
    results['Time'] = results.apply(lambda row: row['Time'] / timem[(row['Dataset'], row['Algorithm'], row['m'])], axis=1)
    results.to_csv('Result/results_real.csv', index = False)
    return results
    
def Post_recruit(saves, S, T, qgrid):
    results = pd.DataFrame(saves.reshape(-1,3))
    results.columns = ['FDP', 'Power', 'Time']
    results['Method'] = np.tile(np.repeat(['cfBH', 'RO-cfBH', 'LOO-cfBH'], len(qgrid)), S*T)
    results['q'] = np.tile(qgrid, S*T*3)
    
    time_means = results.pivot_table(values='Time', index='Method', aggfunc='mean').reset_index()
    results['Time'] = results['Time'] / time_means['Time'][2]
    results.to_csv('Result/results_recruit.csv', index = False)
    return results


def Figure_1(results):
    fig, axs = plt.subplots(3, 4, figsize=(16, 9))

    dgp_algorithms = [('Linear', 'RLM'), ('Linear', 'SGD'), ('Nonlinear', 'RLM'), ('Nonlinear', 'SGD')]
    y_vars = ['Coverage', 'Length', 'Time']
    y_labels = ['Coverage', 'Length', 'Time']
    y_lims = [(0.67, 1.03), (2.2,5.4), (10**-2.7,10**2.7)]
    y_lims_nonlinear = [(0.67, 1.03), (2.5, 5.1), (10**-2.7,10**2.7)]  
    scales = ['linear', 'linear', 'log'] 

    for i, (dgp, algorithm) in enumerate(dgp_algorithms):
        for j, y_var in enumerate(y_vars):
            ax = axs[j, i] 
            sns.boxplot(x='Method', y=y_var, data=results[(results['DGP'] == dgp) & (results['Algorithm'] == algorithm)], ax=ax)

            if dgp == 'Nonlinear':
                if y_lims_nonlinear[j] is not None:
                    ax.set_ylim(y_lims_nonlinear[j])
            else:
                if y_lims[j] is not None:
                    ax.set_ylim(y_lims[j])

            if y_var == 'Coverage':
                ax.hlines(0.9, -0.5, 4.5, colors='r', linestyles='dashed')

            ax.set_yscale(scales[j])

            if j == 0:
                ax.set_title(f'{dgp}: {algorithm}', pad=15)
            if i == 0:
                ax.set_ylabel(y_labels[j], fontsize=18)
            else:
                ax.set_ylabel('')

            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig('Figure/figure_1.png')
    #plt.show()
 
def Figure_2(results, mgrid, variable):
    fig, axs = plt.subplots(len(mgrid), 4, figsize=(16, 9))

    plot_number = 0
    for i, m in enumerate(mgrid):
        for dataset in ['Boston', 'Diabetes']:
            for algorithm in ['RLM', 'SGD']:
                ax = axs[i, plot_number % 4]
                sns.boxplot(x='Method', y=variable, data=results[(results['Algorithm'] == algorithm) & (results['m'] == str(m))], ax=ax)
                
                if i == 0:
                    ax.set_title(f'{dataset}: {algorithm}', pad=15)
                
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                ax.set_xlabel('')
                
                if variable == 'Coverage':
                    ax.hlines(0.9, -0.5, 4.5, colors='r', linestyles='dashed')
                    ax.set_ylim(0.67, 1.03)
                elif variable == 'Time':
                    ax.set_yscale('log')
                
                if plot_number % 4 == 0:
                    ax.set_ylabel(f'm = {m}', fontsize=18)
                else:
                    ax.set_ylabel('')

                plot_number += 1

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig(f'Figure/figure_2_{variable}.png')
    #plt.show()

def Figure_3(results, qgrid):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    sns.boxplot(x='q', y='FDP', hue='Method', data=results, ax=axs[0])
    for x in range(len(qgrid)):
        axs[0].plot([x-0.5, x+0.5], [0.1*(x+1), 0.1*(x+1)], color='red', linestyle='--', linewidth=2)
    axs[0].set_xlabel('q', fontsize=15)
    axs[0].set_ylabel('FDP', fontsize=15)
    axs[0].set_ylim(-0.03, 0.45)

    sns.boxplot(x='q', y='Power', hue='Method', data=results, ax=axs[1])
    axs[1].set_xlabel('q', fontsize=15)
    axs[1].set_ylabel('Power', fontsize=15)

    sns.boxplot(x='Method', y='Time', data=results, ax=axs[2])
    axs[2].set_yscale('log')
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Time', fontsize=15)

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig('Figure/figure_3.png')
    #plt.show()