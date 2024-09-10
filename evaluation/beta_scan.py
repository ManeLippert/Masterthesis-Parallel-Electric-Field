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

kthrho    = 0.3
beta      = [b/1000 for b in range(0,23)]
versions  = ['g', 'f']
markers   = ['D', 's']
mark_size = [10, 6]
colors    = ['#de8f05', '#029e73']

isp, imod, ix = 0, 0, 0

# GROWTH RATE/FREQUENCY TIMETRACE =====

plot.parameters((26,12), 300)

for v,m,s,c in zip(versions, markers, mark_size, colors):
    
    growth_rates = []
    frequencies  = []
    beta_percent = []
    
    picDir = f'../pictures/evaluation/benchmark/{v}-version'
    plot.create_pic_folder(picDir)
    
    for b in beta:
            
        try:
            gamma, omega, time = gkw.beta_scan_data(f'{data_path}/{v}-version/linear/kthrho{kthrho:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
            
            growth_rates.append(gamma[-1])
            frequencies.append(omega[-1])
            beta_percent.append(b*100)
            
        except (FileNotFoundError, BlockingIOError):
            continue
    
    beta_per = []
    gamma, omega = [], []
    
    for b, g, o in zip(beta_percent, growth_rates, frequencies):
        
        if b > 1.0 and b < 1.4:
            beta_per.append(np.nan)
            gamma.append(np.nan)
            omega.append(np.nan)
        
        beta_per.append(b)
        gamma.append(g)
        omega.append(o)

    fig, (ax_growth, ax_freq) = plt.subplots(1, 2, sharex=True)    
    
    ax_growth.plot(beta_per, gamma,m, linestyle='-', color=c, markersize=s)
    ax_freq.plot(beta_per, omega,m, linestyle='-', color=c, markersize=s)
    
    ax_growth.set_ylim(0,1.2)
    ax_freq.set_ylim(0,2.0)
    
    # Horzontal Lines
    ax_growth.plot(np.repeat(1.0, 100), np.linspace(*ax_growth.get_ylim(), 100),linestyle='dashed', color='gray')
    ax_growth.plot(np.repeat(1.2, 100), np.linspace(*ax_growth.get_ylim(), 100),linestyle='dashed', color='gray')
    ax_freq.plot(np.repeat(1.0, 100), np.linspace(*ax_freq.get_ylim(), 100),linestyle='dashed', color='gray')
    ax_freq.plot(np.repeat(1.2, 100), np.linspace(*ax_freq.get_ylim(), 100),linestyle='dashed', color='gray')
    
    # Text
    ax_growth.text(0.5, 0.75, 'ITG', horizontalalignment='center', verticalalignment='center')
    ax_growth.text(1.1, 0.75, 'TEM', horizontalalignment='center', verticalalignment='center')
    ax_growth.text(1.4, 0.75, 'KBM', horizontalalignment='center', verticalalignment='center')
    
    ax_freq.text(0.5, 1.875, 'ITG', horizontalalignment='center', verticalalignment='center')
    ax_freq.text(1.1, 1.875, 'TEM', horizontalalignment='center', verticalalignment='center')
    ax_freq.text(1.4, 1.875, 'KBM', horizontalalignment='center', verticalalignment='center')

    ax_growth.set_xlabel(r'$\beta~[\%]$')
    ax_freq.set_xlabel(r'$\beta~[\%]$')
    ax_growth.set_ylabel(r'$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
    ax_freq.set_ylabel(r'$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')

    plot.postprocessing(fig)
 
    plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{min(beta_percent)/100:.3f}-{max(beta_percent)/100:.3f}_scan_{v}-version.pdf')
    plt.close()