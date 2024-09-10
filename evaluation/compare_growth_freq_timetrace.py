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

picDir = '../pictures/evaluation/benchmark/comparison/timetrace'
plot.create_pic_folder(picDir)

data_path = '../data/benchmark'

# BETA-SCAN ===========================

kthrho   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
beta     = [b/1000 for b in range(0,17)]
versions = ['g','f']
colors   = ['#de8f05', '#029e73']

isp, imod, ix = 0, 0, 0

# GROWTH RATE/FREQUENCY TIMETRACE =====

plot.parameters(32,(12,12), 300)

GVERSION, FVERSION = False, False

for k in kthrho: 
    for b in beta:
        fig, (ax_growth, ax_freq) = plt.subplots(2, 1, sharex=True)
        
        for v, c in zip(versions, colors):
            try:
                gamma, omega, time = gkw.beta_scan_data(f'{data_path}/{v}-version/linear/kthrho{k:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
                
                ax_growth.plot(time, gamma, label = f'{v}-version', color=c)
                ax_freq.plot(time, omega, color = c)
                
                if v == 'g': GVERSION = True
                if v == 'f': FVERSION = True
            
            except (FileNotFoundError, BlockingIOError):
                plt.close()
                continue
        
        ax_freq.set_xlabel(r'$t~[R_\mathrm{ref}/v_{\mathrm{th,ref}}]$')
        ax_growth.set_ylabel(r'$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
        ax_freq.set_ylabel(r'$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
        
        ax_growth.set_xlim(xmax=50)
        
        plot.postprocessing(fig)
        
        if GVERSION and FVERSION:
            
            plt.savefig(f'{picDir}/kthrho{k:.3f}_beta{b:.3f}_growth_freq_timetrace_comparison.pdf') 
            GVERSION, FVERSION = False, False
        
        plt.close()

# GROWTH RATE/FREQUENCY TIMETRACE DIFF

picDir = '../pictures/evaluation/benchmark/comparison/timetrace/difference'
plot.create_pic_folder(picDir)

beta     = [b/1000 for b in range(0,17,2)]
GVERSION, FVERSION = False, False

for k in kthrho: 
    for b in beta:
        fig, (ax_growth, ax_freq) = plt.subplots(2, 1, sharex=True)
        
        try:
            gamma_g, omega_g, time = gkw.beta_scan_data(f'{data_path}/{versions[0]}-version/linear/kthrho{k:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
            
            GVERSION = True
            
        except (FileNotFoundError, BlockingIOError):
            plt.close()
            continue
        
        try:
            gamma_f, omega_f, time = gkw.beta_scan_data(f'{data_path}/{versions[1]}-version/linear/kthrho{k:.3f}/beta{b:.3f}/gkwdata.h5', FIELDS=False)
            
            FVERSION = True
            
        except (FileNotFoundError, BlockingIOError):
            plt.close()
            continue
                
                
                
        ax_growth.plot(time, gamma_g - gamma_f)
        ax_freq.plot(time, omega_g - omega_f)
            
        ax_freq.set_xlabel(r'$t~[R_\mathrm{ref}/v_{\mathrm{th,ref}}]$')
        ax_growth.set_ylabel(r'$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
        ax_freq.set_ylabel(r'$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$')
        
        ax_growth.set_xlim(xmax=50)
        
        plot.postprocessing(fig)
        
        if GVERSION and FVERSION:
            
            plt.savefig(f'{picDir}/kthrho{k:.3f}_beta{b:.3f}_growth_freq_timetrace_diff_comparison.pdf') 
            GVERSION, FVERSION = False, False
        
        plt.close()