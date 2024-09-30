#%% 
# MODULES =============================

# Import modules
import sys, os, h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox

# PATH & IMPORT =======================

sys.path.insert(1, '../python')

import h5tools, plot, gkw

data_path = '../data/benchmark'

# TIMETRACE ===========================

colors   = '#a11a5b'

picDir = f'../pictures/evaluation/benchmark/f-version/fields/timetrace'
plot.create_pic_folder(picDir)

#%%

is_grid = 120
    
start_idx, end_idx = 500, 572
apar, iapar = [], []
try:
    with h5py.File('../data/benchmark/f-version/linear/CBC/kthrho0.027/beta0.008/naverage1_kykxsapar/gkwdata.h5', 'r+') as f:
    
        for i in range(1,572+1):
        
            apar.append(f[f'diagnostic/diagnos_fields/Apa_kykxs00000{i:03}_real'][()][is_grid][0][0])
            iapar.append(f[f'diagnostic/diagnos_fields/Apa_kykxs00000{i:03}_imag'][()][is_grid][0][0])
        
        time = f['diagnostic/diagnos_growth_freq/time'][0]
        s = f["geom/s_grid"][()][is_grid]
        # s = f["geom/s_grid"][()]

        # apar = f[f'diagnostic/diagnos_fields/Apa_kykxs00000{572:03}_real'][()].reshape(288)
        # iapar = f[f'diagnostic/diagnos_fields/Apa_kykxs00000{572:03}_imag'][()].reshape(288)
    
    plot.parameters((12,12), 300)
    
    fig, ax_field = plt.subplots(1, 1, sharex=True)
    
    ax_field.plot(time[start_idx:end_idx], apar[start_idx:end_idx], color = colors, label=r'$\widehat{A}_{1\parallel}^\mathrm{R}$')
    ax_field.plot(time[start_idx:end_idx], iapar[start_idx:end_idx], label=r'$\widehat{A}_{1\parallel}^\mathrm{I}$')
    
    # ax_field.plot(s, apar, color = colors, label=r'$\widehat{A}_{1\parallel}^\mathrm{R}$')
    # ax_field.plot(s, iapar, label=r'$\widehat{A}_{1\parallel}^\mathrm{I}$')
    
    plot.postprocessing(fig)
    
    plt.savefig(f'{picDir}/kthrho0.027_beta0.008_s{s}_t{time[start_idx]}-{time[end_idx-1]}_Apar.pdf')
    
    # plt.close()
    
except (FileNotFoundError, BlockingIOError):
    pass
# %%

# %%
