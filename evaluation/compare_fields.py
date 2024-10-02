# MODULES =============================

# Import modules
import sys, os, h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox

from matplotlib.legend_handler import HandlerBase

# FUNCTIONS ===========================

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           color=orig_handle[0][0], alpha=orig_handle[0][1])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           color=orig_handle[1][0], alpha=orig_handle[1][1])
        return [l1, l2]

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
    
    plot.parameters((26,24), 300, LEGENDONTOP=False, WHITEPLOT = True)
    picDir = f'../pictures/evaluation/benchmark/comparison/fields'
    plot.create_pic_folder(picDir)
    
    fig, ((ax_ar, ax_ai), (ax_er, ax_ei)) = plt.subplots(2, 2, sharex=True)
    
    ax_ar.sharey(ax_er)
    ax_ai.sharey(ax_ei)
    
    i = 1
    
    handles, labels = [], []
    
    for gamma, omega, b, s, a, ia, e, ie in zip(growth_rates, frequencies, beta_percent, s_grid, apar, iapar, epar, iepar): 
        
        ax_ar.plot(s, (-gamma*a + omega*ia), color=colors[0], alpha=i/len(epar))
        # ax_ar.set_xlabel(r'$s$')
        ax_ar.set_ylabel(r'$- \gamma \widehat{A}_{1\parallel}^\mathrm{R} + \omega \widehat{A}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$') #, color=colors[0])
        # ax_ar.tick_params(axis='y', colors=colors[0])
        
        ax_ar.set_title("Real")
        
        ax_er.plot(s, e, color=colors[1], alpha=i/len(epar))
        ax_er.set_xlabel(r'$s$')
        ax_er.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{R}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$') #, color=colors[1])
        # ax_er.tick_params(axis='y', colors=colors[1])
        
        ax_ai.plot(s, (-omega*a -gamma*ia), color=colors[0], alpha=i/len(epar)) #, label=rf'$\beta = {b:.1f}\,\%$')
        # ax_ai.set_xlabel(r'$s$')
        ax_ai.set_ylabel(r'$- \omega \widehat{A}_{1\parallel}^\mathrm{R} - \gamma \widehat{A}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$') #, color=colors[0])
        # ax_ai.tick_params(axis='y', colors=colors[0])
        
        ax_ai.set_title("Imaginary")
        
        ax_ei.plot(s, ie, color=colors[1], alpha=i/len(epar)) #, label=rf'$\beta = {b:.1f}\,\%$')
        ax_ei.set_xlabel(r'$s$')
        ax_ei.set_ylabel(r'$\widehat{E}_{1\parallel}^\mathrm{I}~[2T_\mathrm{ref} \rho_\ast / e R_\mathrm{ref}]$') #, color=colors[1])
        # ax_ei.tick_params(axis='y', colors=colors[1])
        
        handles.append(((colors[0],i/len(epar)), (colors[1], i/len(epar))))
        labels.append(rf'$\beta = {b:.1f}\,\%$')
        
        i += 1
        
    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    
    fig.legend(handles, labels,ncol=1, bbox_to_anchor=(1.1, 0.5), handler_map={tuple: AnyObjectHandler()})
    
    fig.align_labels()
     
    # plot.postprocessing(fig, legcol=1)
        
    # plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{min(beta_percent)/100:.3f}-{max(beta_percent)/100:.3f}_fields_{v}-version.pdf')
    plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{min(beta_percent)/100:.3f}-{max(beta_percent)/100:.3f}_fields_{v}-version.png', transparent = True)
    
    plt.close()

    
    # for b, a, ia, e, ie in zip(beta_percent, apar, iapar, epar, iepar): 
        
    #     plot.parameters((26,12), 300, LEGENDONTOP=False)
    #     fig, ((ax_r, ax_i)) = plt.subplots(1, 2, sharex=True)
        
    #     ax_r.plot(a, color=colors[0])
    #     ax_r.plot(e, color=colors[1])
    #     ax_i.plot(ia, color=colors[0])
    #     ax_i.plot(ie, color=colors[1])
        
    #     fig.subplots_adjust(wspace=0.4, hspace=0.1)
    #     fig.align_labels()
    #     plt.savefig(f'{picDir}/kthrho{kthrho:.3f}_beta{b/100:.3f}_fields_test_{v}-version.pdf')
    #     plt.close()