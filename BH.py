import numpy as np
import pandas as pd

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
    