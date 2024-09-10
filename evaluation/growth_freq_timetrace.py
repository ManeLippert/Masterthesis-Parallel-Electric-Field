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

kthrho   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
beta     = [b/1000 for b in range(0,17)]
versions = ['g', 'f']
colors    = ['#de8f05', '#029e73']

isp, imod, ix = 0, 0, 0

# GROWTH RATE/FREQUENCY TIMETRACE =====

plot.parameters(32,(12,12), 300)

for k in kthrho: 
    for b in beta:
        
        for v,c in zip(versions, colors):
            
            picDir = f'../pictures/evaluation/benchmark/{v}-version/timetrace'
            plot.create_pic_folder(picDir)
            
            try:
                gamma, omega, time = gkw.beta_scan_data(f'{data_path}/{v}-version/linear/kthrho{k:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
                
                fig, (ax_growth, ax_freq) = plt.subplots(2, 1, sharex=True)
                
                ax_growth.plot(time, gamma, color=c)
                ax_freq.plot(time, omega, color=c)
                
                ax_freq.set_xlabel(r'$t~[R_\mathrm{ref}/v_{\mathrm{th,ref}}]$')
                ax_growth.set_ylabel(r'$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
                ax_freq.set_ylabel(r'$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
        
                ax_growth.set_xlim(xmax=50)
        
                plot.postprocessing(fig)
            
                plt.savefig(f'{picDir}/kthrho{k:.3f}_beta{b:.3f}_growth_freq_timetrace_{v}-version.pdf')
            
            except (FileNotFoundError, BlockingIOError):
                continue
        
            plt.close()