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

picDir = '../pictures/evaluation/benchmark'
# Create target Directory if don't exist
if not os.path.exists(picDir):
    os.makedirs(picDir)

# BETA-SCAN ===========================

data_path          = "../data/benchmark/beta_scan"

isp, imod, ix = 0, 0, 0

beta, beta_percent = [], []
growth_rates       = []
frequencies        = []
times              = []
s_grid             = []
apar, iapar        = [], []
epar, iepar        = [], []

GKW = True

for b in range(0,17):

    try: 
        with h5py.File(f"{data_path}/beta{b/1000:.3f}/gkwdata.h5", "r+") as f:

            growth_rate = f["diagnostic/diagnos_growth_freq/growth"][0]
            frequency   = f["diagnostic/diagnos_growth_freq/frequencies"][0]
            time        = f["diagnostic/diagnos_growth_freq/time"][0]
            
            Naverage    = f["input/control/naverage"][0]
            
            parallel_dict = gkw.get_parallel_data(f, DICTIONARY= True)
            
            beta.append(b/1000)
            beta_percent.append(b/10)
            
            if GKW:
                # Only GKW
                growth_rates.append(growth_rate[-1])
                frequencies.append(frequency[-1])
            else:
                # sqrt(2) * GKW = GENE
                growth_rates.append(growth_rate[-1]*np.sqrt(2))
                frequencies.append(frequency[-1]*np.sqrt(2))
    
            times.append(time/Naverage)
    
            s_grid.append(parallel_dict["SGRID"][isp][imod][ix])
            apar.append(parallel_dict["APAR"][isp][imod][ix])
            iapar.append(parallel_dict["iAPAR"][isp][imod][ix])
            epar.append(parallel_dict["EPAR"][isp][imod][ix])
            iepar.append(parallel_dict["iEPAR"][isp][imod][ix])
        
    except (FileNotFoundError, BlockingIOError):
        print(f"gkwdata.h5 for beta{b/1000:.3f} does not exist or is busy")
        
# BETA-SCAN PLOT ======================

plot.parameters(32,(12,12), 300)

fig, (ax_growth, ax_freq) = plt.subplots(2, 1, sharex=True)

ax_growth.plot(beta_percent, growth_rates,"o")
ax_growth.set_ylabel(r"$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$")

ax_freq.plot(beta_percent, frequencies,"o")
ax_freq.set_xlabel(r"$\beta~[\%]$")
ax_freq.set_ylabel(r"$\omega~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$")

fig.suptitle(r"$\beta$ scan in linear simulations")

plt.savefig(f"{picDir}/beta0.000-0.0016_growth_freq.pdf")

# APAR = EPAR PLOT ======================

plot.parameters(32,(12,12), 300)

for s, a, ia, e, ie, b, gamma, omega, t in zip(s_grid, apar, iapar, epar, iepar, beta, growth_rates, frequencies, times):
    
    timestep = t[-1] - t[-2]
    
    fig, ax = plt.subplots(1, 1)

    ax.plot(s, -  a*gamma + ia * omega,"o")
    ax.plot(s, - ia*gamma -  a * omega,"o")
    
    ax.plot(s, e,".")
    ax.plot(s, ie,".")

    # For old implementation    
    # ax.plot(s, e/timestep,".")
    # ax.plot(s, ie/timestep,".")
    
    ax.set_xlabel(r"$s$")
    # ax.set_ylabel(r"$\gamma~[v_{\mathrm{th,ref}}/R_\mathrm{ref}]$")

    fig.suptitle(r"$E_{1\parallel} = \partial_t A_{1\parallel}$ for $\beta=" + f"{b*100 :.1f}" + r"\,\%$")

    plt.savefig(f"{picDir}/beta{b:.3f}_Apar_eq_Epar.pdf")