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
        
            gamma, omega, time, s, a, ia, e, ie = gkw.beta_scan_data(f'{data_path}/{v}-version/linear/CBC/kthrho{kthrho:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=True, isp=isp, imod=imod, ix=ix)
            
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
    
    plot.parameters((30,12), 300)
    picDir = f'../pictures/evaluation/benchmark/{v}-version/fields/adjust_compare'
    plot.create_pic_folder(picDir)
    
    for gamma, omega, b, s, a, ia, e, ie in zip(growth_rates, frequencies, beta_percent, s_grid, apar, iapar, epar, iepar): 

        fig, (ax_r, ax_i) = plt.subplots(1, 2)
        
        ax_r.plot(s, (-gamma*a + omega*ia), linewidth=10, color=colors[0])
        ax_r.set_xlabel(r'$s$')
        ax_r.set_ylabel(r'$- \gamma \widehat{A}_{1\parallel}^\mathrm{R} + \omega \widehat{A}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$', color=colors[0])
        ax_r.tick_params(axis='y', colors=colors[0])
        
        ax_r.set_title("Real")
        
        ax_r_right = ax_r.twinx()
        ax_r_right.sharey(ax_r)
        
        ax_r_right.plot(s, e, color=colors[1])
        ax_r_right.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{R}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$', color=colors[1])
        ax_r_right.tick_params(axis='y', colors=colors[1])
        
        ax_i.plot(s, (-omega*a -gamma*ia), linewidth=10, color=colors[0])
        ax_i.set_xlabel(r'$s$')
        ax_i.set_ylabel(r'$- \omega \widehat{A}_{1\parallel}^\mathrm{R} - \gamma \widehat{A}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$', color=colors[0])
        ax_i.tick_params(axis='y', colors=colors[0])
        
        ax_i.set_title("Imaginary")
        
        ax_i_right = ax_i.twinx()
        ax_i_right.sharey(ax_i) 
        
        ax_i_right.plot(s, ie, color=colors[1])
        ax_i_right.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$', color=colors[1])
        ax_i_right.tick_params(axis='y', colors=colors[1])
            
        # fig.suptitle(rf'$\beta = {b}\,\%$')
        fig.subplots_adjust(wspace=0.7)
        
        plot.postprocessing(fig)
        
        plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{b/100:.3f}_fields_{v}-version.pdf')
        plt.close()