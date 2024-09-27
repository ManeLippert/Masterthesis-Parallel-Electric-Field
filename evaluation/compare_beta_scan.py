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

# BETA-SCAN ===========================

kthrho   = 0.3
beta     = [b/1000 for b in range(0,23)]
versions = ['g', 'f']
markers  = ['D', 's']
colors   = ['#de8f05', '#029e73']

isp, imod, ix = 0, 0, 0

# GROWTH RATE/FREQUENCY TIMETRACE =====

plot.parameters((25,12), 300)
    
picDir = f'../pictures/evaluation/benchmark/comparison/growth_rate_freq'
plot.create_pic_folder(picDir)

fig, (ax_growth, ax_freq) = plt.subplots(1, 2, sharex=True)

growth_rates = []
frequencies  = []
beta_percent = []

for b in beta:
            
    try:
        
        
        gamma, omega, time = gkw.beta_scan_data(f'{data_path}/{versions[0]}-version/linear/CBC/kthrho{kthrho:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
            
        growth_rates.append(gamma[-1])
        frequencies.append(omega[-1])
        beta_percent.append(b*100)
            
    except (FileNotFoundError, BlockingIOError):
        continue

beta_per = []
gamma, omega = [], []
    
for b, g, o in zip(beta_percent, growth_rates, frequencies):
        
    # if b > 1.0 and b < 1.4:
    #     beta_per.append(np.nan)
    #     gamma.append(np.nan)
    #     omega.append(np.nan)
        
    if b > 1.1 and b < 1.4:
        beta_per.append(np.nan)
        gamma.append(np.nan)
        omega.append(np.nan)
        
    beta_per.append(b)
    gamma.append(g)
    omega.append(o)

ax_growth.plot(beta_per, gamma, markers[0], linestyle='-', label=f'{versions[0]}-version', color=colors[0], markersize=10)
ax_freq.plot(beta_per, omega, markers[0], linestyle='-', color=colors[0], markersize=10)

growth_rates = []
frequencies  = []
beta_percent = []

for b in beta:
    
    try: 
        
        gamma, omega, time = gkw.beta_scan_data(f'{data_path}/{versions[1]}-version/linear/CBC/kthrho{kthrho:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
            
        growth_rates.append(gamma[-1])
        frequencies.append(omega[-1])
        beta_percent.append(b*100)
            
    except (FileNotFoundError, BlockingIOError):
        continue
    
beta_per = []
gamma, omega = [], []
    
for b, g, o in zip(beta_percent, growth_rates, frequencies):
        
    # if b > 1.0 and b < 1.4:
    #     beta_per.append(np.nan)
    #     gamma.append(np.nan)
    #     omega.append(np.nan)
        
    if b > 1.1 and b < 1.4:
        beta_per.append(np.nan)
        gamma.append(np.nan)
        omega.append(np.nan)
        
    beta_per.append(b)
    gamma.append(g)
    omega.append(o)
    
ax_growth.plot(beta_per, gamma, markers[1], linestyle='-', label=f'{versions[1]}-version', color=colors[1])
ax_freq.plot(beta_per, omega, markers[1], linestyle='-', color=colors[1])

ax_growth.set_ylim(0,1.2)
ax_freq.set_ylim(0,2.0)
    
 # Horzontal Lines
# ax_growth.plot(np.repeat(1.0, 100), np.linspace(*ax_growth.get_ylim(), 100),linestyle='dashed', color='gray')
# ax_growth.plot(np.repeat(1.2, 100), np.linspace(*ax_growth.get_ylim(), 100),linestyle='dashed', color='gray')
# ax_freq.plot(np.repeat(1.0, 100), np.linspace(*ax_freq.get_ylim(), 100),linestyle='dashed', color='gray')
# ax_freq.plot(np.repeat(1.2, 100), np.linspace(*ax_freq.get_ylim(), 100),linestyle='dashed', color='gray')
    
# Text
ax_growth.text(0.6, 1.2*0.95, 'ITG', horizontalalignment='center', verticalalignment='center')
# ax_growth.text(1.1, 1.2*0.95, 'TEM', horizontalalignment='center', verticalalignment='center')
ax_growth.text(1.75, 1.2*0.95, 'KBM', horizontalalignment='center', verticalalignment='center')
    
ax_freq.text(0.6, 2.0*0.95, 'ITG', horizontalalignment='center', verticalalignment='center')
# ax_freq.text(1.1, 2.0*0.95, 'TEM', horizontalalignment='center', verticalalignment='center')
ax_freq.text(1.75, 2.0*0.95, 'KBM', horizontalalignment='center', verticalalignment='center')

ax_growth.set_xlabel(r'$\beta~[\%]$')
ax_freq.set_xlabel(r'$\beta~[\%]$')
ax_growth.set_ylabel(r'$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
ax_freq.set_ylabel(r'$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')

fig.subplots_adjust(wspace=0.3)

plot.postprocessing(fig)
 
plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{min(beta):.3f}-{max(beta):.3f}_scan_comparison.pdf')
plt.close()