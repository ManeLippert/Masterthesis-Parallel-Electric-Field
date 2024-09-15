import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.transforms import Bbox
import seaborn as sns
from cycler import cycler
import os

import warnings

warnings.filterwarnings("error")

# Plot parameters
def parameters(figsize, dpi, fontsize = 40, USETEX = True,
               linewidth = 3, colorpalette = sns.color_palette("colorblind", as_cmap=True),
               tickwidth = 1.5, ticklength = 10, tickspad = 10, ticksdirc = 'in', TICKSADDITIONAL = True,
               legendpad = -2, LEGENDONTOP = True, WHITEPLOT = False):
    
    # TEXT ==========================================================
    
    plt.rcParams['text.usetex'] = USETEX
    plt.rcParams['font.family'] = 'serif'
    
    # fontsize = fontsize or 
    
    plt.rcParams['font.size'] = fontsize
    
    # FIGURE ========================================================
    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    
    # STYLE =========================================================
    
    plt.rcParams['axes.labelpad'] = 15
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.prop_cycle'] = cycler('color', colorpalette)
    
    # PLOT MARGINS ==================================================
    
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    
    # TICKS AND FRAME ===============================================
    
    plt.rcParams['axes.linewidth'] = tickwidth
    
    plt.rcParams['xtick.top'] = TICKSADDITIONAL
    plt.rcParams['xtick.bottom'] = TICKSADDITIONAL
    
    plt.rcParams['ytick.left'] = TICKSADDITIONAL
    plt.rcParams['ytick.right'] = TICKSADDITIONAL
    
    plt.rcParams['xtick.major.size'] = ticklength
    plt.rcParams['xtick.major.width'] = tickwidth
    
    plt.rcParams['ytick.major.size'] = ticklength
    plt.rcParams['ytick.major.width'] = tickwidth
    
    plt.rcParams['xtick.minor.size'] = ticklength/2
    plt.rcParams['xtick.minor.width'] = tickwidth
    
    plt.rcParams['ytick.minor.size'] = ticklength/2    
    plt.rcParams['ytick.minor.width'] = tickwidth
    
    plt.rcParams['xtick.direction'] = ticksdirc
    plt.rcParams['ytick.direction'] = ticksdirc
    
    plt.rcParams['xtick.major.pad'] = tickspad
    plt.rcParams['ytick.major.pad'] = tickspad
    
    # LEGEND ========================================================
    
    plt.rcParams['legend.frameon'] = False
    
    if LEGENDONTOP:
        plt.rcParams['legend.loc'] = 'upper center'
        plt.rcParams['legend.borderpad'] = 0
        # plt.rcParams['legend.borderaxespad'] = 0
        #plt.rcParams['legend.handleheight'] = 1
        # plt.rcParams['legend.borderpad'] = legendpad
    else:
        plt.rcParams['legend.loc'] = 'center right'
        plt.rcParams['legend.borderpad'] = 1
        
    # WHITE PLOT ====================================================
    
    if WHITEPLOT:
        plt.rcParams['xtick.color']='white'
        plt.rcParams['ytick.color']='white'
        plt.rcParams['axes.labelcolor']='white'
        plt.rcParams['axes.edgecolor']='white'
        plt.rcParams['lines.color']='white'
        plt.rcParams['text.color']='white'
        
    # SAVE FIG ======================================================
    
    plt.rcParams['savefig.bbox'] = 'tight'
    
def ax_ticks_subplot(ax):
    ax.tick_params(direction = "in")

    axT = ax.secondary_xaxis('top')
    axT.tick_params(direction = "in")
    axT.xaxis.set_ticklabels([])

    axR = ax.secondary_yaxis('right')
    axR.tick_params(direction = "in")
    axR.yaxis.set_ticklabels([])

# Bbox-Size for saving subplots
def full_extent(ax, pad):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def savefig_subplot(fig, ax, path, pad, bbox_input = None):
    #extent = full_extent(ax, pad).transformed(fig.dpi_scale_trans.inverted())
    if bbox_input == None:
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    else:
        bbox = bbox_input
        
    try:
        bbox = bbox.expanded(1.0 + pad[1], 1.0 + pad[0])
    except TypeError:
        bbox = bbox.expanded(1.0 + pad, 1.0 + pad)
        
    fig.savefig(path, bbox_inches=bbox)
    
    
def postprocessing(fig, legcol = 5):
    
    fig.align_labels()
    
    try:
        fig.legend(ncol=legcol)
    except UserWarning:
        pass

def create_pic_folder(picDir):

    # Create target Directory if don't exist
    if not os.path.exists(picDir):
        os.makedirs(picDir)