#Import modules
import numpy as np
import pandas as pd
import h5py
import h5tools
import derivative

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox

# From Florian Rath
def get_eflux_time(hdf5_file):
    
    # find out number of time steps
    time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
    nt = time.shape[1]
    
    
    # name of the dataset
    node_name = 'diagnostic/diagnos_fluxes/eflux_species01'
    
    # load data into array
    data = np.transpose(hdf5_file[node_name][()])
    
    # reshape gkw flux ordering
    flux = np.reshape(data,(nt,2))[:,0]
    
    return flux, time[0]

# From Florian Rath
def get_radial_density_profile(hdf5_file, start = None, end = None):

    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

    try:
        dr_dens = hdf5_file[h5tools.find_key(hdf5_file, 'derivative_dens')][()]
        
    except TypeError:
        print('Calculation of density derivative...')
        
        dens = hdf5_file['diagnostic/diagnos_moments/dens_kyzero_xs01'][()]
        time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
        time = time[0]

        rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
        rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

        # Mean over s
        zonal_dens = np.mean(dens,0)

        nx = zonal_dens.shape[0]
        dx = rad_boxsize/nx

        dr_dens = - derivative.finite_first_order(zonal_dens[:,:], dx, 'central', PERIODIC=True)

    # Mean over t
    dr_dens_mean = np.mean(dr_dens[:,start:end], 1)
    
    return dr_dens, dr_dens_mean, rad_coord

def get_second_radial_density_derivative(hdf5_file, start = None, end = None):

    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

    try:
        ddr_dens = hdf5_file[h5tools.find_key(hdf5_file, 'second_derivative_dens')][()]
        
    except TypeError:
        print('Calculation of second density derivative...')
        
        dens = hdf5_file['diagnostic/diagnos_moments/dens_kyzero_xs01'][()]
        time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
        time = time[0]

        rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
        rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

        # Mean over s
        zonal_dens = np.mean(dens,0)

        nx = zonal_dens.shape[0]
        dx = rad_boxsize/nx

        ddr_dens = derivative.finite_second_order(zonal_dens[:,:], dx, 'period')

    # Mean over t
    ddr_dens_mean = np.mean(ddr_dens[:,start:end], 1)
    
    return ddr_dens, ddr_dens_mean, rad_coord

def get_radial_energy_profile(hdf5_file, start = None, end = None):
    
    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]
    
    try:
        dr_ene = [hdf5_file[h5tools.find_key(hdf5_file, 'derivative_energy_perp')][()], hdf5_file[h5tools.find_key(hdf5_file, 'derivative_energy_par')][()]]
    
    except (TypeError, ValueError):
        print('Calculation of radial energy derivative...')
    
        ene = [hdf5_file['diagnostic/diagnos_moments/ene_perp_kyzero_xs01'][()], hdf5_file['diagnostic/diagnos_moments/ene_par_kyzero_xs01'][()]]

        time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
        time = time[0]

        rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
        rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

        # Mean over s
        zonal_ene = [np.mean(ene[0],0), np.mean(ene[1],0)]
        dr_ene = []

        for i in zonal_ene:
        
            nx = i.shape[0]
            dx = rad_boxsize/nx

            dr = - derivative.finite_first_order(i[:,:], dx, 'central', PERIODIC=True)
            
            dr_ene.append(dr)

    dr_ene_mean = []

    for i in dr_ene:
        
        # Mean over t
        dr_mean = np.mean(i[:,start:end], 1)

        dr_ene_mean.append(dr_mean)

    return dr_ene, dr_ene_mean, rad_coord

def get_second_radial_energy_derivative(hdf5_file, start = None, end = None):
    
    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]
    
    try:
        ddr_ene = [hdf5_file[h5tools.find_key(hdf5_file, 'second_derivative_energy_perp')][()], hdf5_file[h5tools.find_key(hdf5_file, 'second_derivative_energy_par')][()]]
    
    except (TypeError, ValueError):
        print('Calculation of second radial energy derivative...')
    
        ene = [hdf5_file['diagnostic/diagnos_moments/ene_perp_kyzero_xs01'][()], hdf5_file['diagnostic/diagnos_moments/ene_par_kyzero_xs01'][()]]

        time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
        time = time[0]

        rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
        rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]

        # Mean over s
        zonal_ene = [np.mean(ene[0],0), np.mean(ene[1],0)]
        ddr_ene = []

        for i in zonal_ene:
        
            nx = i.shape[0]
            dx = rad_boxsize/nx

            ddr = derivative.finite_second_order(i[:,:], dx, 'period')
            
            ddr_ene.append(ddr)

    ddr_ene_mean = []

    for i in ddr_ene:
        
        # Mean over t
        ddr_mean = np.mean(i[:,start:end], 1)

        ddr_ene_mean.append(ddr_mean)

    return ddr_ene, ddr_ene_mean, rad_coord

def get_radial_zonal_potential_profile(hdf5_file, start, end):
    
    # Stepsize
    rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]
    
    try:
        dr_zonal_pot = hdf5_file[h5tools.find_key(hdf5_file, 'derivative_zonalflow_potential')][()]
                
    except TypeError:
        print('Calculation of potential derivative...')
        
        time = hdf5_file['diagnostic/diagnos_growth_freq/time'][()]
        time = time[0]
        
        try: 
            zonal_pot = hdf5_file[h5tools.find_key(hdf5_file, 'zonalflow_potential')][()]
            dx = hdf5_file[h5tools.find_key(hdf5_file, 'derivative_stepsize')][()]
            
        except TypeError:
            
            # Elektrostatic potencial
            phi = hdf5_file[h5tools.find_key(hdf5_file, 'phi')][:,:]
            nx = phi.shape[0]
    
            dx = rad_boxsize/nx
        
            # Mean over y to get a approximation for the zonal potenzial
            zonal_pot = np.mean(phi,1)

        dr_zonal_pot = derivative.finite_first_order(zonal_pot[:,:], dx, 'central', PERIODIC=True)

    # Mean over t
    dr_zonal_pot_mean = np.mean(dr_zonal_pot[:,start:end], 1)
    
    return dr_zonal_pot, dr_zonal_pot_mean, rad_coord


def get_shearingrate_radialcoordinate_radialboxsize_ddphi_dx_zonalpot(hdf5_file, start_index = None, end_index = None):
    
    # Stepsize
    rad_boxsize = hdf5_file[h5tools.find_key(hdf5_file, 'lxn')][()][0]
    rad_coord = hdf5_file[h5tools.find_key(hdf5_file,'xphi')][0,:]
    
    try:
        wexb =  hdf5_file[h5tools.find_key(hdf5_file, 'shearing_rate')][()]
        zonal_pot = hdf5_file[h5tools.find_key(hdf5_file, 'zonalflow_potential')][()]
        ddphi = hdf5_file[h5tools.find_key(hdf5_file, 'second_derivative_phi')][()]
        dx = hdf5_file[h5tools.find_key(hdf5_file, 'derivative_stepsize')][()]
        #print('Loaded from data')
        
    except TypeError:
        print('Calculation shearing rate...')
        
        # Elektrostatic potencial
        phi = hdf5_file[h5tools.find_key(hdf5_file, 'phi')][:,:,start_index:end_index]
        nx = phi.shape[0]
    
        dx = rad_boxsize/nx
        
        # Mean over y to get a approximation for the zonal potenzial
        zonal_pot = np.mean(phi,1)

        # Finite Difference for shearing rate omega_ExB

        ddphi= derivative.finite_second_order(zonal_pot[:,:], dx, 'period')
        wexb = 0.5 * ddphi
    
    return wexb, rad_coord, rad_boxsize, ddphi, dx, zonal_pot

def get_max_shearingrate(hdf5_file, wexb, time, fourier_index_max):
    
    try:
        #print('Loaded from data')
        wexb_max =  hdf5_file[h5tools.find_key(hdf5_file, 'shearing_rate_maximum')][()]
        
    except TypeError:
        print('Calculation...')
        def wexb_max_data(fourier_index_max):

            data = []

            for time_point in range(len(time)):

                wexb_time_point = wexb[:,time_point]
                wexb_fft = np.fft.fft(wexb_time_point)
                wexb_fft_amp = 2/len(wexb_fft) * np.abs(wexb_fft)
                data.append(wexb_fft_amp[fourier_index_max])

            return data

        wexb_max = []

        # Calculate wexb_max as list of list with multiple index    
        for i in range(fourier_index_max+1):
            wexb_max.append(wexb_max_data(i))

        wexb_max = np.array(wexb_max)
    
    return wexb_max

def get_index_from_value(data, value):
    n, index = 0, None
    
    for i in data:
        
        if round(i) == value:
            index = n
        n += 1

    return index

def get_mean_middle_shearingrate(start, end, wexb):
    middle = int((end - start)/2 + start)

    # Shearing rate with mean over time
    wexb_rad_mean = np.mean(wexb[:,start:end],1)
    wexb_rad_middle = wexb[:,middle]
    
    return wexb_rad_mean, wexb_rad_middle

def get_fft_mean_max_shearingrate_amplitude(wexb_mean):
    wexb_rad_mean_fft = np.fft.fft(wexb_mean)
    wexb_rad_mean_amp = 2/ wexb_rad_mean_fft.shape[0] * np.abs(wexb_rad_mean_fft)
    wexb_rad_mean_amp_max = max(wexb_rad_mean_amp)
    
    return wexb_rad_mean_amp, wexb_rad_mean_amp_max

def get_data_info(df_path, boxsize, 
                  finit = 'cosine2', 
                  Ns = 16, Nvpar = 48, Nmu = 9, 
                  rlt = 6.0,
                  dtim = 0.020, krhomax = 1.4):
    
    df = pd.read_csv(df_path)
    
    df = df.loc[df['rlt'] == rlt]
    df = df.loc[df['boxsize'] == boxsize]
    df = df.loc[df['Ns'] == Ns]
    df = df.loc[df['Nvpar'] == Nvpar]
    df = df.loc[df['Nmu'] == Nmu]
    df = df.loc[df['dtim'] == dtim]
    df = df.loc[df['krhomax'] == krhomax]
    df = df.loc[df['finit'] == finit]
    
    return df

# Returns the data from parallel data organized by indexes
# as numpy array or dictionary with column namesa as key
#
#   - icol = index column
#   - isp  = index species
#   - imod = index binormal gird
#   - ix   = index radial grid
#
# Note that the index starts as always in python at zero!
# If multipile runs gets performed the last data set will 
# be extracted
#
# numpy array: parallel_array[ical][isp][imod][ix]
# dictionary : parallel_dict[key][isp][imod][ix]
def get_parallel_data(hdf5_file, DICTIONARY = False):
    
    Nx   = int(hdf5_file['input/grid/n_x_grid'][0])
    Ns   = int(hdf5_file['input/grid/n_s_grid'][0])
    Nmod = int(hdf5_file['input/grid/nmod'][0]) 
    Nsp  = int(hdf5_file['input/grid/number_of_species'][0])
    
    parallel_data = hdf5_file['diagnostic/diagnos_mode_struct/parallel'][:, - Nx*Ns*Nmod*Nsp:]
    parallel_key  = ['SGRID', 
                     'PHI', 'iPHI', 
                     'APAR', 'iAPAR', 
                     'DENS', 'iDENS',
                     'ENE_PAR', 'iENE_PAR',
                     'ENE_PERP', 'iENE_PERP',
                     'WFLOW', 'iWFLOW',
                     'BPAR', 'iBPAR',
                     'EPAR', 'iEPAR']
    
    parallel_Nsp, parallel_Nmod, parallel_Nx = [], [], []
    
    for icol in range(len(parallel_data)):
        split_Nsp = np.reshape(parallel_data[icol], (Nsp, Nmod*Nx*Ns))
        
        for isp in range(len(split_Nsp)):
            split_Nmod = np.reshape(split_Nsp[isp], (Nmod, Nx*Ns))
            
            for imod in range(len(split_Nmod)):
                split_Nx = np.reshape(split_Nmod[imod], (Nx, Ns))
                parallel_Nx.append(split_Nx)
            
            parallel_Nmod.append(np.array(parallel_Nx))
            parallel_Nx = []
        
        parallel_Nsp.append(np.array(parallel_Nmod))
        parallel_Nmod = []

    parallel_array = np.array(parallel_Nsp)
    
    if DICTIONARY:
        parallel_dict = {}
        for k,v in zip(parallel_key, parallel_array):
            parallel_dict[k] = v
        
        return parallel_dict
    else:
        return parallel_array
    
def beta_scan_data(hdf5_path, GENE = False, FIELDS = True,
                   isp = 0, imod = 0, ix = 0):

    with h5py.File(hdf5_path, 'r+') as f:

        gamma = f['diagnostic/diagnos_growth_freq/growth'][0]
        omega = f['diagnostic/diagnos_growth_freq/frequencies'][0]
        time  = f['diagnostic/diagnos_growth_freq/time'][0]

        Naverage    = f['input/control/naverage'][0]

        parallel_dict = get_parallel_data(f, DICTIONARY= True)

        if GENE:
            # sqrt(2) * GKW = GENE
            gamma = gamma*np.sqrt(2)
            omega = omega*np.sqrt(2)

        if FIELDS:
            s_grid = (parallel_dict['SGRID'][isp][imod][ix])
            apar   = (parallel_dict['APAR'][isp][imod][ix])
            iapar  = (parallel_dict['iAPAR'][isp][imod][ix])
            epar   = (parallel_dict['EPAR'][isp][imod][ix])
            iepar  = (parallel_dict['iEPAR'][isp][imod][ix])
            
            return gamma, omega, time, s_grid, apar, iapar, epar, iepar
        else:
            return gamma, omega, time
        

# PLOT FUNCTIONS ==================================================================================

def eflux_time(time, eflux, figuresize = (24,8), xlim = (0, None), ylim = (0, None), label = None, create_plot = True, axis = None):
    
    if create_plot:
        fig, axis = plt.subplots(figsize=figuresize)
    
    axis.plot(time, eflux, label = label)
    axis.set_xlabel(r'$t~[R/ \nu_{\mathrm{th}}]$')
    axis.set_ylabel(r'$\chi~[\rho^2 \nu_{\mathrm{th}} / R]$')
    
    if xlim[1] == None:
        axis.set_xlim((0, max(time)))
    else:
        axis.set_xlim(xlim)
    
    axis.set_ylim(ylim)
    
def max_shearingrate_time(time, wexb_max, fourier_index, figuresize):
    fig, ax = plt.subplots(figsize=figuresize)
    
    if type(fourier_index) == int:
        ax.plot(time,wexb_max[fourier_index], label = r'$k_' + str(fourier_index) + '$')
    else:
        for i in fourier_index:
            ax.plot(time,wexb_max[i], label = r'$k_' + str(i) + '$')

    ax.set_xlabel(r'$t~[R/ \nu_{\mathrm{th}}]$')
    ax.set_ylabel(r'$|\widehat{\omega}_{\mathrm{E \times B}}|_{n_\mathrm{ZF}}~[\nu_{\mathrm{th}}/R]$')
    
    ax.set_xlim(xmin=0, xmax=time[-1])
    ax.set_ylim(ymin=0)
    
    ax_ticks_subplot(ax)
    
    plt.legend(ncol = max(fourier_index))
    
def all_shearingrate_radialcoordinate(rad_coord, wexb, figuresize, stepsize):
    fig, ax = plt.subplots(figsize=figuresize)

    start, end = 0, stepsize - 1 
    
    while end <= wexb.shape[1]:
        
        wexb_mean = np.mean(wexb[:,start:end],1)
    
        ax.plot(rad_coord, wexb_mean)
    
        start += stepsize
        end += stepsize
    
    #ax.set_title(r'$R/L_T =$ ' + rlt + ', time interval [0 '+str(wexb.shape[1])+']', pad=20)
    ax.set_xlabel(r'$\psi[\rho]$')
    ax.set_ylabel(r'$\omega_{\mathrm{E \times B}}$')
    
    ax.set_xlim(xmin=0, xmax=rad_coord[-1])
    ax.set_ylim(ymin=-0.45, ymax=0.45)
    
    ax_ticks_subplot(ax)
    
def mean_shearingrate_radialcoordinate_amplitude(rad_coord, wexb_rad_mean, wexb_rad_middle, wexb_rad_mean_amp, wexb_rad_mean_amp_max, 
                                                 figuresize):
    
    if figuresize == (24,8):
        fig, ax = plt.subplots(1, 2, figsize=figuresize)

        # Plot shearing rate
        ax[0].plot(rad_coord, wexb_rad_mean)
        ax[0].plot(rad_coord, wexb_rad_middle, 'black', linestyle='--', linewidth=1)
        ax[0].plot(rad_coord, np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
        ax[0].plot(rad_coord, -np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
        #ax[0].set_title(r'$R/L_T =$ ' + rlt + ', time interval [' + str(start) + ' ' + str(end) + ']', pad=20)
        ax[0].set_xlabel(r'$\psi[\rho]$')
        ax[0].set_ylabel(r'$\omega_{\mathrm{E \times B}}$')
        
        ax[0].set_xlim(xmin=0)

        ax_ticks_subplot(ax[0])

        #savefig_subplot(fig, ax[0], '../pictures/'+data+'/'+path+'/'+data+'_'+resolution+'_wexb_'+str(start)+'_'+str(end)+'.pdf', 0.02)

        # FT{shearing rate}
        ax[1].plot(rad_coord[1:], wexb_rad_mean_amp[1:])
        #ax[1].set_title(r'$R/L_T =$ ' + rlt + ', time interval [' + str(start) + ' ' + str(end) + ']', pad=20)
        ax[1].set_xlabel(r'$\psi[\rho]$')
        ax[1].set_ylabel(r'Amplitude')
        
        ax[1].set_xlim(xmin=0)
        
        ax_ticks_subplot(ax[1])

        #savefig_subplot(fig, ax[1],'../pictures/'+data+'/'+path+'/'+data+'_'+resolution+'_Amp_Rad_'+str(start)+'_'+str(end)+'.pdf', 0.02)
    elif figuresize == (12,8):
        fig, ax = plt.subplots(figsize=figuresize)
        
        # Plot shearing rate
        ax.plot(rad_coord, wexb_rad_mean)
        ax.plot(rad_coord, wexb_rad_middle, 'black', linestyle='--', linewidth=1)
        ax.plot(rad_coord, np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
        ax.plot(rad_coord, -np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
        #ax.set_title(r'$R/L_T =$ ' + rlt + ', time interval [' + str(start) + ' ' + str(end) + ']', pad=20)
        ax.set_xlabel(r'$\psi[\rho]$')
        ax.set_ylabel(r'$\omega_{\mathrm{E \times B}}$')
        
        ax.set_xlim(xmin=0)
        
        ax_ticks_subplot(ax)
      
def mean_shearingrate_radialcoordinate_subplot(rad_coord, rad_boxsize, wexb_rad_mean, wexb_rad_middle, wexb_rad_mean_amp_max, 
                                               ax, x, y, x_max, y_max, start_time, end_time):
    
    if y_max > 1:
        axis = ax[y,x]
    else:
        if x_max > 1:
            axis = ax[x]
        else:
            axis = ax

    # Plot shearing rate
    axis.plot(rad_coord, wexb_rad_mean)
    #axis.plot(rad_coord, wexb_rad_middle, 'black', linestyle='--', linewidth=1)
    axis.plot(rad_coord, np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
    axis.plot(rad_coord, -np.repeat(wexb_rad_mean_amp_max, len(rad_coord)), 'r', linestyle='--', linewidth=1)
    #ax.set_title(r'$R/L_T =$ ' + rlt + ', time interval [' + str(start) + ' ' + str(end) + ']', pad=20)
    #ax[y,x].set_xlabel(r'$x[\rho]$')
    #ax[y,x].set_ylabel(r'$\omega_{\mathrm{E \times B}}$')
    
    y_axis_height = 0.45
    
    axis.set_ylim([-y_axis_height, y_axis_height])
    axis.set_xlim([0, rad_boxsize])
    
    ax_ticks_subplot(axis)
    
    #amp_label_height_neg = (y_axis_height - wexb_rad_mean_amp_max)/(2*y_axis_height) - 0.1
    amp_label_height_neg = 0.05
    amp_label_neg=r'$-\,A_{\mathrm{max}}$'
    
    amp_label_height_pos = 1 - amp_label_height_neg
    amp_label_pos=r'$A_{\mathrm{max}}$ = ' + format(round(wexb_rad_mean_amp_max, 2), '.2f')
    
    title = 'time interval [' + str(start_time) + ', ' + str(end_time) + ']'
    
    axis.text(0.805, amp_label_height_neg, amp_label_neg, color='r', ha='center', va='center', transform=axis.transAxes)
    axis.text(0.87, amp_label_height_pos, amp_label_pos, color='r', ha='center', va='center', transform=axis.transAxes)
    axis.text(0.5, 0.95, title, ha='center', va='center', transform=axis.transAxes)
    
def dim_subplot(interval):
    
    for i in [3,2,1]:
        if interval.shape[1] % i == 0:
            xdim, ydim = i, int(interval[1]/i)
            break
