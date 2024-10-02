import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def post_synthetic(saves, T):
    results = pd.concat([pd.DataFrame(saves[0].reshape(-1, 3)), \
                        pd.DataFrame(saves[1].reshape(-1, 3)), \
                        pd.DataFrame(saves[2].reshape(-1, 3)), \
                        pd.DataFrame(saves[3].reshape(-1, 3))], axis = 0)

    results.columns = ['Coverage', 'Length', 'Time']
    results['DGP'] = np.repeat(['Linear', 'Nonlinear'], T * 5 * 2)
    results['Method'] = np.tile(['OracleCP', 'FullCP', 'SplitCP', 'RO-StabCP', 'LOO-StabCP'], T * 2 * 2)
    results['Algorithm'] = np.tile(np.repeat(['RLM', 'SGD'], T * 5), 2)
    results['LogTime'] = np.log10(results['Time'])
    
    results.to_csv('Result/results_synthetic.csv', index = False)
    return results

def post_real(saves, T, mgrid):
    results = pd.concat([pd.DataFrame(saves[0].reshape(-1, 3)), \
                        pd.DataFrame(saves[1].reshape(-1, 3)), \
                        pd.DataFrame(saves[2].reshape(-1, 3)), \
                        pd.DataFrame(saves[3].reshape(-1, 3))], axis = 0)
    results.columns = ['Coverage', 'Length', 'Time']
    results['Dataset'] = np.repeat(['Boston', 'Diabetes'], T * 5 * len(mgrid) * 2)
    results['Method'] = np.tile(np.repeat(['OracleCP', 'FullCP', 'SplitCP', 'RO-StabCP', 'LOO-StabCP'], len(mgrid)), T * 2 * 2)
    results['Algorithm'] = np.tile(np.repeat(['RLM', 'SGD'], T * 5 * len(mgrid)), 2)
    results['m'] = np.tile(mgrid, T * 5 * 2 * 2).astype(str)
    results['LogTime'] = np.log10(results['Time'])
    
    results.to_csv('Result/results_real.csv', index = False)
    return results
    
def post_recruit(saves, T, qgrid):
    results = pd.DataFrame(saves.reshape(-1,3))
    results.columns = ['FDP', 'Power', 'Time']
    results['Method'] = np.tile(np.repeat(['cfBH', 'RO-cfBH', 'LOO-cfBH'], len(qgrid)), T)
    results['q'] = np.tile(qgrid, T*3)
    results['LogTime'] = np.log10(results['Time'])
    
    results.to_csv('Result/results_recruit.csv', index = False)
    return results


def figure_1(results):
    fig, axs = plt.subplots(3, 4, figsize=(16, 9))

    dgp_algorithms = [('Linear', 'RLM'), ('Linear', 'SGD'), ('Nonlinear', 'RLM'), ('Nonlinear', 'SGD')]
    y_vars = ['Coverage', 'Length', 'LogTime']
    y_labels = ['Coverage', 'Length', 'Log-time (log(sec.))']
    y_lims = [(0.66, 1.03), None, None] 
    y_lims_nonlinear = [(0.66, 1.03), None, None]  
    scales = ['linear', 'linear', 'linear'] 

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
            bold_font = fm.FontProperties(family='Times New Roman', weight='bold', size=12)
            for label in ax.get_xticklabels():
                if label.get_text() == 'LOO-StabCP':
                    label.set_fontproperties(bold_font)

            if j == 0:
                ax.set_title(f'{dgp} Setting: {algorithm}', pad=15)
            if i == 0:
                ax.set_ylabel(y_labels[j], fontsize=18)
            else:
                ax.set_ylabel('')

            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig('Figure/figure_1.png')
    plt.show()
 
def figure_2(results, mgrid):
    datasets_algorithms = [('Boston', 'RLM'), ('Boston', 'SGD'), ('Diabetes', 'RLM'), ('Diabetes', 'SGD')]
    y_vars = ['Length', 'LogTime']
    y_labels = ['Length', 'Log-time (log(sec.))']
    length_lim = [(2.1,4.3),(1.3,2.6),(2.45,3.3),(2.05,2.95)]
    time_lim = (-2.3, 2.3)

    bold_font = fm.FontProperties(weight='bold')

    fig, axs = plt.subplots(2 * len(mgrid), 4, figsize=(16, 6 * len(mgrid)))

    for i, (dataset, algorithm) in enumerate(datasets_algorithms):
        for j, y_var in enumerate(y_vars):
            for k, m in enumerate(mgrid):
                ax = axs[2 * j + k, i]
                sns.boxplot(x='Method', y=y_var, data=results[(results['Algorithm'] == algorithm) & (results['Dataset'] == dataset) & (results['m'] == str(m))], ax=ax)
                if m == 1:
                    ax.text(0.96, 0.935,'    =1',fontsize=14, bbox=dict(facecolor='white', edgecolor='black', linewidth=0.5),transform=ax.transAxes,
                        horizontalalignment='right',verticalalignment='top')
                    ax.text(0.89, 0.94,'m',fontstyle='italic',fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0),transform=ax.transAxes,
                        horizontalalignment='right',verticalalignment='top')
                else:
                    ax.text(0.96, 0.935,'    =100',fontsize=14, bbox=dict(facecolor='white', edgecolor='black', linewidth=0.5),transform=ax.transAxes,
                        horizontalalignment='right',verticalalignment='top')
                    ax.text(0.83, 0.94,'m',fontstyle='italic',fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0),transform=ax.transAxes,
                        horizontalalignment='right',verticalalignment='top')


                if y_var == 'Length':
                    if dataset == 'Boston':
                        if algorithm == 'RLM':
                            ax.set_ylim(length_lim[0])
                        else:
                            ax.set_ylim(length_lim[1])
                    if dataset == 'Diabetes':
                        if algorithm == 'RLM':
                            ax.set_ylim(length_lim[2])
                        else:
                            ax.set_ylim(length_lim[3])
                else:
                    ax.set_ylim(time_lim)
                            
                        
                for label in ax.get_xticklabels():
                    if label.get_text() == 'LOO-StabCP':
                        label.set_fontproperties(bold_font)

                if (j == 0) and (k == 0):
                    ax.set_title(f'{dataset}: {algorithm}', pad=15)
                        
                ax.set_xlabel('')
                if i == 0:
                    if y_var == 'Length':
                        ax.set_ylabel(y_labels[0], fontsize=18)
                    else:
                        ax.set_ylabel(y_labels[1], fontsize=18)
                else:
                    ax.set_ylabel('')
                        
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig('Figure/figure_2.png')
    plt.show()

def figure_3(results, qgrid):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    sns.boxplot(x='q', y='FDP', hue='Method', data=results, ax=axs[0])
    for x in range(len(qgrid)):
        axs[0].plot([x-0.5, x+0.5], [0.1*(x+1), 0.1*(x+1)], color='red', linestyle='--', linewidth=2)
    axs[0].set_xlabel('q', fontsize=18, fontstyle='italic')
    axs[0].set_ylabel('FDP', fontsize=18)
    axs[0].set_xlim(-0.48, 2.48)
    axs[0].set_ylim(-0.03, 0.54)

    sns.boxplot(x='q', y='Power', hue='Method', data=results, ax=axs[1])
    axs[1].set_xlabel('q', fontsize=18, fontstyle='italic')
    axs[1].set_ylabel('Power', fontsize=18)

    sns.boxplot(x='Method', y='LogTime', data=results, ax=axs[2])
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Log-time (log(sec.))', fontsize=18)

    for legend_text in axs[0].get_legend().get_texts():
        if legend_text.get_text() == 'LOO-cfBH':
            legend_text.set_fontproperties(fm.FontProperties(family='Times New Roman', weight='bold', size=10))
    for legend_text in axs[1].get_legend().get_texts():
        if legend_text.get_text() == 'LOO-cfBH':
            legend_text.set_fontproperties(fm.FontProperties(family='Times New Roman', weight='bold', size=10))

    for label in axs[2].get_xticklabels():
        if label.get_text() == 'LOO-cfBH':
            label.set_fontproperties(fm.FontProperties(family='Times New Roman', weight='bold', size=15))

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.savefig('Figure/figure_3.png')
    plt.show()