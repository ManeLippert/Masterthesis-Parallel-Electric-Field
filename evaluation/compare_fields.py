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
colors   = ['#a11a5b', '#0173b2']

isp, imod, ix = 0, 0, 0

# GROWTH RATE/FREQUENCY TIMETRACE =====

for v in versions:
    
    growth_rates = []
    frequencies  = []
    beta_percent = []
    s_grid       = []
    apar, iapar  = [], []
    epar, iepar  = [], []
    
    for b in beta:
        
        try: 
        
            gamma, omega, time, s, a, ia, e, ie = gkw.beta_scan_data(f'{data_path}/{v}-version/linear/kthrho{kthrho:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=True, isp=isp, imod=imod, ix=ix)
            
            growth_rates.append(gamma[-1])
            frequencies.append(omega[-1])
            beta_percent.append(b*100)
            s_grid.append(s)
            apar.append(a)
            iapar.append(ia)
            epar.append(e)
            iepar.append(ie)
            
        except (FileNotFoundError, BlockingIOError):
            continue
    
    plot.parameters((26,24), 300)
    picDir = f'../pictures/evaluation/benchmark/comparison'
    plot.create_pic_folder(picDir)
    
    fig, ((ax_ar, ax_ai), (ax_er, ax_ei)) = plt.subplots(2, 2)
    
    i = 1
    
    for gamma, omega, b, s, a, ia, e, ie in zip(growth_rates, frequencies, beta_percent, s_grid, apar, iapar, epar, iepar): 
        
        ax_ar.plot(s, (-gamma*a + omega*ia)*1e3, color=colors[0], alpha=i/len(epar))
        ax_ar.set_xlabel(r'$s$')
        ax_ar.set_ylabel(r'$- \gamma \widehat{A}_{1\parallel}^\mathrm{R} + \omega \widehat{A}_{1\parallel}^\mathrm{I}$', color=colors[0])
        ax_ar.tick_params(axis='y', colors=colors[0])
        
        ax_er.plot(s, e*1e3, color=colors[1], alpha=i/len(epar))
        ax_er.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{R}$', color=colors[1])
        ax_er.tick_params(axis='y', colors=colors[1])
        
        ax_ai.plot(s, (-omega*a -gamma*ia)*1e3, color=colors[0], alpha=i/len(epar))
        ax_ai.set_xlabel(r'$s$')
        ax_ai.set_ylabel(r'$- \omega \widehat{A}_{1\parallel}^\mathrm{R} - \gamma \widehat{A}_{1\parallel}^\mathrm{I}$', color=colors[0])
        ax_ai.tick_params(axis='y', colors=colors[0])
        
        ax_ei.plot(s, ie*1e3, color=colors[1], alpha=i/len(epar))
        ax_ei.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{I}$', color=colors[1])
        ax_ei.tick_params(axis='y', colors=colors[1])
        
        # fig.suptitle(rf'$\beta = {b}\,\%$')
        
        i += 1
        
    fig.subplots_adjust(wspace=0.5)
        
    plot.postprocessing(fig)
        
    plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_fields_{v}-version.pdf')
    plt.close()