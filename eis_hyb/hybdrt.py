''' This module contains functions to format, plot, analyze, and save EIS and DRT spectra for stability testing.
The data comes from a Gamry Potentiostat
Code heavily adapted from code written by Dr. Jake Huang
As of now this script can only be used in the hybdrt conda environment
# C-Meisel
'''

'Imports'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import re
import cmasher as cmr
import matplotlib.ticker as ticker
import matplotlib as mpl

from .data_formatting import * 
from .convenience import rp_ohmic_to_excel, df_tau_r_to_excel, plot_drt_peaks, find_cell_name, quick_dualdrt_plot
from .stability import plot_r_over_time

import hybdrt
import hybdrt.fileload as fl
import hybdrt.plotting as hplt
from hybdrt.models import DRT
from hybdrt.mapping import DRTMD

import zmaptools
import zmaptools.dataload as dl
from zmaptools import clean
from zmaptools import fit as zfit
import zmaptools.plot as zplt

from hybdrt.utils import stats
from hybdrt import mapping
from hybdrt import filters as hf
from hybdrt.utils.array import nearest_index


' -- Ascetic stuff'
# data_kw = dict(facecolors='none', edgecolors=[0.1] * 3)
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['font.sans-serif'] = 'Arial'

' ----- HybDRT functions'
def plot_hybdrt(cp_file:str,z_file:str,**plot_args):
    '''
    Plots Hybdrt data
    Takes a chronopotentiometry (cp_file) data file and an EIS data file (z_file)
    
    Parameters:
    -----------
    cp_file, str:
        path to the chronopotentiometry file
    z_file, str:
        path to the eis file

    Return --> Nothing, but a plot is made and shown
    '''
    # Make a tupple of the chronopotentiometry and EIS data
    hy_tup = fl.get_hybrid_tuple(cp_file, z_file)

    hy_drt = DRT() # Create a DRT instance

    # # -- Fit the hybrid data
    # -- You can enable/disable background estimation by passing subtract_background=True or False
    hy_drt.fit_hybrid(*hy_tup, subtract_background=False, background_type='dynamic',
                    background_corr_power=0.5)

    # Plot the hybrid results
    hy_drt.plot_results(**plot_args)
    plt.show()

def plot_hybdrts(cp_file:str,z_file:str,**plot_args):
    '''
    Plots Hybdrt data
    Takes a chronopotentiometry (cp_file) data file and an EIS data file (z_file)
    
    Parameters:
    -----------
    cp_file, str:
        path to the chronopotentiometry file
    z_file, str:
        path to the eis file

    Return --> Nothing, but a plot is made and shown
    '''
    # Make a tupple of the chronopotentiometry and EIS data
    hy_tup = fl.get_hybrid_tuple(cp_file, z_file)

    hy_drt = DRT() # Create a DRT instance

    # # -- Fit the hybrid data
    # -- You can enable/disable background estimation by passing subtract_background=True or False
    hy_drt.fit_hybrid(*hy_tup, subtract_background=False, background_type='dynamic',
                    background_corr_power=0.5)

    # Plot the hybrid results
    hy_drt.plot_distribution(**plot_args)

' ----- Mapping functions'
def fit_polmap(data_path:str, jar:str, pattern:str, pkl_name = '', area:float = 0.5, stb:bool = False):
    '''
    Fits a singular pol-map
    Code pretty much copy and pasted from Dr. Jake Huang

    Parameters:
    ------------
    data_path, str:
        path to where the pol map datafiles are stored in your computer
    jar, str:
        path to where the pol map fit should be stored
    pattern, str:
        unique identifier for the pol_map
        If this is left blank or set to None, all staircase files in the directory will be loaded
    area, float: (default = 0.5)
        active cell area in cm^2

    Return --> Nothing, but the data is fit and saved
    '''
    # - Setting path strs to paths:
    data_path = Path(data_path)
    jar = Path(jar)

    # # -- Set up multi-dimensional DRT (DRTMD)
    # Specify the variables that differentiate distinct pol maps - these are up to you
    # I've put the variables that I used as a placeholder
    psi_dim_names = ['T', 'po2', 'ph2', 'ph2o'] # These are different testing parameters for the test

    # Add variables that differentiate invididual measurements within each pol map
    # These will be determined from the data files
    psi_dim_names = psi_dim_names + ['j', 'V', 'eta', 'time']

    # Create tau supergrid - this is the array of possible grid points used to fit each measurement
    tau_supergrid = np.logspace(-7.5, 2.5, 101)

    # Create basis_nu for DOP fits
    basis_nu = np.concatenate([np.linspace(-1, -0.4, 25), np.linspace(0.4, 1, 25)])

    # Create the multi-dimensional DRT instance
    mrt = DRTMD(tau_supergrid=tau_supergrid, 
                fit_dop=True, fixed_basis_nu=basis_nu, nu_basis_type='gaussian',
                fit_type='drt', warn=False, psi_dim_names=psi_dim_names)

    # Specify which ideal circuit elements to include (only ohmic should be included if fit_dop=True)
    mrt.fit_ohmic = True
    mrt.fit_inductance = False
    mrt.fit_capacitance = False

    # Set the keyword arguments to use when calling fit_hybrid (or fit_eis)
    # These are my recommended settings, but you may want to play around with them.
    mrt.fit_kw = dict(
        # Allow negative DRT values - usually important for fitting electrolysis spectra
        nonneg=False, 
        # Estimate and remove the chrono background.
        subtract_background=True, background_type='dynamic', background_corr_power=0.5,
        estimate_background_kw={'length_scale_bounds': (0.05, 5), 'n_restarts': 2, 'noise_level_bounds': (0.01, 10)},
        # Find and remove extreme values
        remove_extremes=True,
        # Use an outlier-robust error structure
        remove_outliers=False, outlier_p=0.05, #outlier_thresh=0.75,
    )

    # specify the variables that are constant during the pol map.
    # these correspond to the first four psi_dim_names specified in the cell above (T, po2, ph2, ph2o).
    # I've put common values in here since I don't know the true measurement conditions.
    constant_psi_vals = [550, 0.21, 1.0, 0.03]

    # Specify the names of any variables that should be extracted from the DTA file NOTES section.
    # A value of None indicates no metadata to be extracted.
    # In the future, I would recommend setting up your scripts to save metadata in the NOTES section
    # so that you aren't reliant on file names or lookups to determine the conditions for each measurement.
    psi_note_fields = None

    # Set the initial timestamp to which all measurements will be referenced.
    if len(np.unique(mrt.obs_group_id)) == 0:
        ocv_file = next(data_path.glob('OCP_*.DTA'))
        init_timestamp = fl.get_timestamp(ocv_file)

    else:
        init_timestamp = init_ocp_timestamp(data_path,pattern)

    # Load and fit the pol map files.
    # It should take 1-5 seconds to fit each file. 
    # Using background estimation takes longer (more like 3-5 s instead of < 1 s per file), but it's generally a good idea.
    # Progress will be printed. You may also see diagnostic messages and warnings, but these are normal.
    zfit.fit_pmap_data(mrt, data_path, pattern, constant_psi_vals, psi_note_fields, 
                       init_time=init_timestamp,
                       area=area,
                       # Setting prefer_filtered=True indicates that the filtered chrono files will be used 
                       # instead of the unfiltered files
                       prefer_filtered=True
                      )

    # The data and fits are assigned to groups.
    # In general, a group should correspond to a single pol map (i.e. one set of conditions).
    # The filename pattern will be the group ID
    for group_id in np.unique(mrt.obs_group_id):
        print('Group {} contains {} observations'.format(group_id, len(mrt.get_group_index(group_id))))
        
    # Save to file
    if pkl_name == '':
        mrt.save_attributes('all', jar.joinpath(f'{pattern}.pkl'))
    else:
        mrt.save_attributes('all', jar.joinpath(f'{pkl_name}.pkl'))

def plot_polmap(polmap, group_by = None,structure_plot = False,normalize_plot = False,
                all_ridge_trough = False, save_pm_r:str = None, save_pm_p:str=None,
                title:str = None, publication:bool = False):
    '''
    Plots an already fit polmap
    Code pretty much Copy-pasted from code written from Dr. Jake Huang

    Parameters:
    ----------
    polmap, str:
        path to the .pkl file saving the polmap fit
    group_by, str: (default = None)
        The parameter to group the pol maps by if the .pkl file has
        multiple pol maps in it
        ex: 'time', 'po2', 'T', etc...
    structure_plot, bool: (default = False)
        If True, plots the structure plot
    normalize_plot, bool: (default = False)
        If True, plots the normalized pol map
    all_ridge_trough, bool: (default = False)
        If True, plots all ridge and trough plots
    save_pm_r, str: (default = None)
        If this is not none, the resistance pol-map will be saved.
        save_pm_r is the file name and path of the saved file.
    save_pm_t, str: (default = None)
        If this is not none, the probability pol-map will be saved.
        save_pm_t is the file name and path of the saved file.
    title, str: (default = None)
        If not none, the string title is the title printed on the plot
    publication, bool: (default = False)
        If false the figure is formatted for a presentation
        If true the figure is formatted to be a subfigure in a journal paper.
        Setting publication to true increases all font sizes


    Return --> None, but plots and shows a pol map
    '''
    mrt = DRTMD.from_source(polmap) # Load File

    ' ------- Make overpotential (eta) grid '
    # Get measured eta values
    etas = np.sort(mrt.obs_psi_df['eta'].values)

    # - Get median step size
    v_step = np.median(np.diff(etas))

    # - Make uniform grid based on median step size
    steps = np.round(etas / v_step - 0.5, 0)
    v_base = (steps + 0.5) * v_step

    #  - Visually confirm that grid aligns with measured values
    fig, ax = plt.subplots()
    ax.plot(v_base, v_base, c='k', label='ideal', alpha=0.5)
    ax.scatter(v_base, etas, s=40, alpha=0.5, label='measured')

    ax.set_xlabel('v_grid (V)')
    ax.set_ylabel('Measured $\eta$ (V)')
    ax.legend()
    fig.tight_layout()
    plt.show()

    ' - - - - - Get pol map "structure" - - - - - '
    v_base = np.unique(v_base) # 
    # Here we take the individual DRT fits, refine and aggregate them, 
    # and apply filters and other functions for analysis

    # If you had multiple groups in the drtmd object, you would want to group by the relevant variables.
    # Since we have only one pol map, we don't need to do any grouping
    if group_by is None:
        struct_results = zmaptools.structure.structure_from_drtmd(mrt, v_base, group_by=[], #'po2', 'ph2', 'ph2o'],
                                                                flag_kw={'thresh': 2},
                                                            remove_bad=True, resolve=True, impute=True)
    else:
        struct_results = zmaptools.structure.structure_from_drtmd(mrt, v_base, group_by=[group_by], #'po2', 'ph2', 'ph2o'],
                                                                flag_kw={'thresh': 2},
                                                            remove_bad=True, resolve=True, impute=True)        

    # The struct_results output will have one entry for each temperature.
    # I'm just extracting the 550 C results for convenience
    struct_550 = struct_results[550]

    if structure_plot == True:
        ' ----- First, lets plot the resulting 2D DRT. -----'
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

        # Here we plot several different versions of the 2D DRT:
        # x_raw: the raw (individual) DRT fits, which are quite noisy
        # x_res: the DRT fits after applying a batch fitting procedure to refine the DRT 
        # by fitting sections of the surface to multiple measurements simultaneously
        # x_filt: the filtered surface, obtained by applying filters and outlier removal to x_res
        # You can see that the structure becomes clearer with each refinement step
        for i, name in enumerate(['x_raw', 'x_res', 'x_filt']):
            zplt.plot_drt_2d(v_base, struct_550[name], mrt, ax=axes[i])
            axes[i].set_title(name)
            
        # Red indicates positive DRT values, blue indicates negative DRT values.

    ' ----- Plotting map with a colorbar'
    fig, ax = plt.subplots()
    sm, ax = zplt.plot_drt_2d(v_base, struct_550['x_filt'], mrt, ax=ax)
    cbar = fig.colorbar(sm, ax=ax, label=r'$\gamma$ ($\Omega$)',pad=0.025)
    tick_fs = 21
    title_fs = 25

    # - Excessive formatting
    if publication == False:
        ax.tick_params(axis='both', which='major', labelsize='x-large') #changing tick label size
        ax.xaxis.get_label().set_fontsize('xx-large')
        ax.yaxis.get_label().set_fontsize('xx-large')
        cbar.ax.set_ylabel(r'$\gamma$ ($\Omega$)', fontsize='xx-large')
        cbar.ax.tick_params(axis='y', labelsize='x-large')

    if publication == True:
        ax.tick_params(axis='both', which='major', labelsize=tick_fs, width=2, length=6) #changing tick label size
        ax.xaxis.get_label().set_fontsize(title_fs)
        ax.yaxis.get_label().set_fontsize(title_fs)
        cbar.ax.set_ylabel(r'$\gamma$ ($\Omega$)', labelpad=-2, fontsize=title_fs)
        cbar.ax.tick_params(axis='y', labelsize=tick_fs, width=2, length=6)
    
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    # - hline and text
    hline_color = '#4c4c4c'
    if publication == False:
        ax.text(2.25, 0.04, 'EC', color=hline_color, ha='right', va='center',
                weight='bold',size='xx-large')
        ax.text(2.25, -0.04, 'FC', color=hline_color, ha='right', va='center',
                weight='bold',size='xx-large')
        ax.axhline(0, color = hline_color, linewidth=1.5)
    else:
        fs = 26
        ax.text(2.25, 0.02, 'EC', color=hline_color, ha='right', va='bottom',
            weight='bold',size=title_fs)
        ax.text(2.25, -0.03, 'FC', color=hline_color, ha='right', va='top',
                weight='bold',size=title_fs)
        ax.axhline(0, color = hline_color, linewidth=2.5)   

    # - Adding a title:
    if title is not None:
        plt.title(title, fontsize=28, weight='bold')
    
    fig.tight_layout()

    # - Saving Pol-map
    if save_pm_r is not None:
        fmat = save_pm_r.split('.', 1)[-1]
        fig.savefig(save_pm_r, dpi=300, format=fmat, bbox_inches='tight')
    plt.show()

    if normalize_plot == True:
        ' ----- Normalizing the DRT to Rp at each voltage ----- '
        # This makes it easier to see the structure even when the Rp is very small relative to other parts of the surface
        fig, ax = plt.subplots(figsize=(3.5, 3))
        # Specify normalize=True to normalize to Rp
        sm, ax = zplt.plot_drt_2d(v_base, struct_550['x_filt'], mrt, ax=ax, normalize=True)
        fig.colorbar(sm, ax=ax, label=r'$\gamma \, / \, R_p$ ($\Omega$)')
        fig.tight_layout()
        plt.show()

    ' ----- The next step is to look for the processes (ridges) in the structure ----- ' 
    if all_ridge_trough == True:
        # There ridge probability function (rp_raw) and trough probability function (tp_raw)
        # are helpful for highlighting possible ridges and troughs in between ridges
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

        for i, name in enumerate(['x_filt', 'rp_raw', 'tp_raw']):
            if name  == 'x_filt':
                cmap = 'coolwarm'
                # plot_x_2d will determine vlimits
                vmin = None
                vmax = None
            else:
                cmap = 'viridis'
                # Probability functions are always in the interval (0, 1)
                vmin = 0
                vmax = 1
            zplt.plot_x_2d(v_base, struct_550[name], mrt, ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(name)
            
            # tp_raw is mostly helpful as complementary information if rp_raw isn't totally clear

            fig.tight_layout()
            plt.show()
            
    else: # Just plot the ridge plot
        fig, ax = plt.subplots()
        cmap = 'viridis'
        vmin = 0
        vmax = 1
        name = 'rp_raw'
        zplt.plot_x_2d(v_base, struct_550[name], mrt, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        # - Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([vmin,vmax])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(sm,ticks = [],cax=cax)
        
        if publication == False:
            cb.set_label(label='Probability',fontsize = 'xx-large')

            # - hline and text
            hline_color = '#cccccc'
            ax.text(2.25, 0.04, 'EC', color=hline_color, ha='right', va='center',
                    weight='bold',size='xx-large')
            ax.text(2.25, -0.04, 'FC', color=hline_color, ha='right', va='center',
                    weight='bold',size='xx-large')
            ax.axhline(0, color = hline_color, linewidth=1.5)

            # - Excessive formatting
            ax.tick_params(axis='both', which='major', labelsize='x-large') #changing tick label size
            ax.xaxis.get_label().set_fontsize('xx-large')
            ax.yaxis.get_label().set_fontsize('xx-large')

        if publication == True:
            cb.set_label(label='Probability',fontsize = title_fs)

            # - hline and text
            hline_color = '#cccccc'
            ax.text(2.25, 0.02, 'EC', color=hline_color, ha='right', va='bottom',
                    weight='bold',size=title_fs)
            ax.text(2.25, -0.03, 'FC', color=hline_color, ha='right', va='top',
                    weight='bold',size=title_fs)
            ax.axhline(0, color = hline_color, linewidth=2.5)

            # - Excessive formatting
            ax.tick_params(axis='both', which='major', labelsize=tick_fs, width=2, length=6) #changing tick label size
            ax.xaxis.get_label().set_fontsize(title_fs)
            ax.yaxis.get_label().set_fontsize(title_fs)

        fig.tight_layout()

        # - Saving Pol-map
        if save_pm_p is not None:
            fmat = save_pm_p.split('.', 1)[-1]
            fig.savefig(save_pm_p, dpi=300, format=fmat, bbox_inches='tight')
        plt.show()

        fig.tight_layout()
        plt.show()

' --- Adjacent plotting functions'
def plot_stb_holds(data_path:str, area:float = 0.5, clean_axis:bool=True, quant_stb:str = 'all',
    first_file:str = 'default', save_fig:str = None, publication:bool = False, **plotargs:dict):
    '''
    Plotting the data from the potentiostatic holds or ocv holds during a test

    Parameters:
    ------------
    data_path, str:
        path to where the pol map datafiles are stored
    area, float: (default = 0.5)
        active area of the cell
    clean_axis, string: (default = True)
        When true, plots the Voltage axis with less points
        This is a much cleaner way to plot and looks better when presenting
        however it clears out the Y axis so if you want to zoom in on an image, set this to False
    quant_stb, str: (default = 'all')
        This is how the stability data gets reported on the graph. There are many ways to quantify stability
        if = mv, the slope of the graph is multiplied by 1,000,000 to report the data in mV or mA/khrs
        if = percent, then the slope is multiplied by 1,000,000 then divided by the starting potential/current to get %/khrs 
        (% of the starting potential lost during testing)
        if = all, then all three of the above options are printed at on the figure
    first_file, str" (default = 'default')
        finds the first ocv file to attain the time the test started
        if set to 'default' the init ocp file is used as the beginning time
        if there is a specific file that should be first, set first_file to the path to that file
    save_fig, str: (default = None)
        If this is not none, the galvano and ocv holds figure is  saved.
        save_fig is the file name and path of the saved file.
    plotargs, dict:
        Any arguments that are passed to the plot function.
        if this is None, then the markersize is set to 15
    publication, bool: (default = False)
        If false the figure is formatted for a presentation
        If true the figure is formatted to be a subfigure in a journal paper.
        Setting publication to true increases all feature sizes
    '''
    ' -_-_-_- Attaining Data'
    # --- Attaining all relevant files
    pstat_files = find_files_with_string(data_path, 'PSTATIC')
    ocv_files = find_files_with_string(data_path, 'OCP')
    ocv_files = [s for s in ocv_files if 'Kst' not in s]

    # ---  Attaining the initial Values
    # - Timestamp
    if first_file == 'default':
        start_ocv_file = [s for s in ocv_files if 'PolMap-Init-.DTA' in s]
        start_ocv_path = Path(start_ocv_file[0])
        t0_stamp = fl.get_timestamp(start_ocv_path)
        t0 = t0_stamp.strftime("%s")
    else:
        start_ocv_path = first_file
        t0_stamp = fl.get_timestamp(first_file)
        t0 = int(t0_stamp.strftime("%s"))

    # - OCV
    df_start_ocv = get_ocv_data(start_ocv_path)
    start_ocv = df_start_ocv['V vs. Ref.'].mean()

    # - Current
    first_pstat_file = [s for s in pstat_files if 'PolMap-1-.DTA' in s]
    first_pstat_path = Path(first_pstat_file[0])
    df_first_pstat = get_ocv_data(first_pstat_path)
    start_current = df_first_pstat['A'].head(360).mean() / area * -1 # Area specific starting current in A/cm^2

    # --- Making dataframes with all the hold data
    # - Potentiostatic hold data
    all_pstat = concatenate_data(pstat_files,data_type='pot')
    all_pstat['s'] = (all_pstat['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    all_pstat['A'] = all_pstat['A'] / area * -1 # Converting from i to j. ohm/cm^2.
    all_pstat.to_excel('/Users/Charlie/Documents/CSM/pstat.xlsx')

    pstat_start_time = int(all_pstat['s'].min())
    pstat_end_time = int(all_pstat['s'].max())

    # # - OCV hold data
    all_ocv = concatenate_data(ocv_files,data_type='ocv')
    all_ocv['s'] = (all_ocv['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    
    ocv_start_time = int(all_ocv['s'].min())
    ocv_end_time = int(all_ocv['s'].max())

    ' -_-_-_- Plotting'
    # --- Plot setup
    fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8)) # Default figure size is 6.4*4.8

    if len(plotargs)==0 and publication == False: # Basically if the dictionary is empty
        plotargs['markersize'] = 15
    elif len(plotargs)==0 and publication == True:
        plotargs['markersize'] = 30

    # --------- Plotting
    axs[0].plot(all_pstat['s'],all_pstat['A'],'.k',**plotargs)
    axs[1].plot(all_ocv['s'],all_ocv['V vs. Ref.'],'.k',**plotargs)

    # ---- General Formatting
    if publication == False:
        fontsize = 18
        fit_linewidth=1
        y_label_pad = -35
        x_label_pad = -15
        spine_width = 1
    else:
        fontsize = 26
        txt_size = fontsize*0.85
        fit_linewidth=2
        y_label_pad = 0
        x_label_pad = -20
        spine_width = 2

    # ------- Pstat Plot formatting
    axs[0].set_xlabel('Time (hrs)',fontsize = fontsize)
    axs[0].set_ylabel('Current at 0.8V (A/cm$^2$)',fontsize = fontsize)
    
    # --- Pstat Excessive formatting
    axs[0].set_xticks([pstat_start_time,pstat_end_time])
    
    if clean_axis == True:
        axs[0].set_yticks([0,start_current])
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        axs[0].yaxis.labelpad = y_label_pad
    
    if publication == False:
        axs[0].set_xticklabels([pstat_start_time,pstat_end_time],fontsize=fontsize-2)
        axs[0].yaxis.set_label_coords(-0.035, 0.45)
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize-2,width=1,length=3) #changing tick label size

    else:
        axs[0].set_xticklabels([pstat_start_time,pstat_end_time],fontsize=txt_size)
        axs[0].tick_params(axis='both', which='major', labelsize=txt_size,
                            width=spine_width,length=spine_width*3) #changing tick label size

    axs[0].xaxis.labelpad = x_label_pad 
    axs[0].spines['bottom'].set_bounds(pstat_start_time, pstat_end_time)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_ylim(0,start_current*1.2)
    for spine in axs[0].spines.values():
        spine.set_linewidth(spine_width)

    # --- Pstat fitting and writing slope on graph:
    m_ps,b_ps = np.polyfit(all_pstat['s'],all_pstat['A'],1)
    fit = m_ps*all_pstat['s']+b_ps
    axs[0].plot(all_pstat['s'],fit,'--r',linewidth=fit_linewidth)
    x0_center = np.mean(axs[0].get_xlim())
    y0_range = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    if publication == False:
        decimals = 2
    elif publication == True:
        decimals=0
    
    if quant_stb == 'mv':
        mp_ps = m_ps * 1000000 # Converting the slope into a mA per khrs (*1000 to get from A to mA, *1000 to get to khrs,*-1 for degradation)
        ms_ps = f'{round(mp_ps,2)}'
        axs[0].text(x0_center,y0_range*0.085, ms_ps+' mA/khrs',weight='bold',size='xx-large', ha='center', va='center')
    
    if quant_stb == 'percent':
        mp_ps = ((m_ps * 1000)/(start_current))*100 # Converting the slope into a mA per khrs = * 1000, dividing by init_A to get stb/khrs, and multiply by 100 to get %/khrs)
        ms_ps = f'{round(mp_ps,2)}'
        axs[0].text(x0_center,y0_range*0.085, ms_ps+' %/khrs',weight='bold',size='xx-large', ha='center', va='center')
    
    if quant_stb == 'all':
        m_ma = m_ps * 1000000 # Converting the slope into a mA per khrs (*1000 to get from A to mA, *1000 to get to khrs,*-1 for degradation)
        m_mar = round(m_ma,decimals)
        m_mas = "{:.{}f}".format(m_mar, decimals)

        m_amp = ((m_ps * 1000)/(start_current))*100 # Converting the slope into a mA per khrs = * 1000, dividing by init_A to get stb/khrs, and multiply by 100 to get %/khrs)
        m_ampr = round(m_amp,decimals)
        m_amps = "{:.{}f}".format(m_ampr, decimals)

        if publication == False:
            axs[0].text(x0_center,y0_range*0.12, m_mas + ' mA/khrs',weight='bold',size='xx-large', ha='center', va='center')
            axs[0].text(x0_center,y0_range*0.05, m_amps + ' %/khrs',weight='bold',size='xx-large', ha='center', va='center')
        elif publication == True:
            axs[0].text(x0_center,y0_range*0.20, m_mas + ' mA/khrs',weight='bold',size=fontsize, ha='center', va='center')
            axs[0].text(x0_center,y0_range*0.08, m_amps + ' %/khrs',weight='bold',size=fontsize, ha='center', va='center')
    
    # ------- OCV Plot formatting
    axs[1].set_xlabel('Time (hrs)',fontsize = fontsize)
    axs[1].set_ylabel('OCV (V)',fontsize = fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=txt_size) #changing tick label size
    axs[1].set_xticks([ocv_start_time,ocv_end_time])

    # --- OCV Excessive formatting
    axs[1].set_xticks([ocv_start_time,ocv_end_time])
    
    if clean_axis == True:
        axs[1].set_yticks([0,start_ocv])
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        axs[1].yaxis.labelpad = y_label_pad
    
    if publication == False:
        axs[1].set_xticklabels([ocv_start_time,ocv_end_time],fontsize=fontsize-2)
    else:
        axs[1].set_xticklabels([ocv_start_time,ocv_end_time],fontsize=txt_size)
        axs[1].yaxis.labelpad = x_label_pad*2.5


    axs[1].xaxis.labelpad = x_label_pad
    axs[1].spines['bottom'].set_bounds(ocv_start_time, ocv_end_time)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_ylim(0,start_ocv*1.2)
    for spine in axs[1].spines.values():
        spine.set_linewidth(spine_width)

    # --- OCV fitting and writing slope on graph:
    m_oc,b_oc = np.polyfit(all_ocv['s'],all_ocv['V vs. Ref.'],1)
    fit = m_oc*all_ocv['s']+b_oc
    axs[1].plot(all_ocv['s'],fit,'--r',linewidth=fit_linewidth)
    x1_center = np.mean(axs[1].get_xlim())
    y1_range = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]
    
    if quant_stb == 'mv':
        mp_oc = m_oc * 1000000 # Converting the slope into a mV per khrs (*1000 to get from A to mA, *1000 to get to khrs,*-1 for degradation)
        ms_oc = f'{round(mp_oc,2)}'
        axs[1].text(x1_center,y1_range*0.085, ms_oc+' mV/khrs',weight='bold',size='xx-large', ha='center', va='center')
    
    if quant_stb == 'percent':
        mp_oc = ((m_oc * 1000)/(start_ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
        ms_oc = f'{round(mp_oc,2)}'
        axs[1].text(x1_center,y1_range*0.085, ms_oc+' %/khrs',weight='bold',size='xx-large', ha='center', va='center')
    
    if quant_stb == 'all':
        m_ocv = m_oc * 1000000 # Converting the slope into a mV per khrs (*1000 to get from A to mA, *1000 to get to khrs,*-1 for degradation)
        m_ocvs = f'{round(m_ocv,2)}'

        m_ocp = ((m_oc * 1000)/(start_ocv))*100 # Converting the slope into a mV per khrs = * 1000, dividing by init_v to get stb/khrs, and multiply by 100 to get %/khrs)
        m_ocps = f'{round(m_ocp,2)}'

        if publication == False:
            axs[1].text(x1_center,y1_range*0.12, m_ocvs + ' mV/khrs',weight='bold',size='xx-large', ha='center', va='center')
            axs[1].text(x1_center,y1_range*0.05, m_ocps + ' %/khrs',weight='bold',size='xx-large', ha='center', va='center')
        elif publication == True:
            axs[1].text(x1_center,y1_range*0.20, m_ocvs + ' mV/khrs',weight='bold',fontsize=fontsize, ha='center', va='center')
            axs[1].text(x1_center,y1_range*0.08, m_ocps + ' %/khrs',weight='bold',fontsize=fontsize, ha='center', va='center')          

    plt.tight_layout()
    
    if save_fig is not None: # - Saving the figure
        fmat = save_fig.split('.', 1)[-1]
        fig.savefig(save_fig, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

def plot_stb_near_ocv_drt(data_path, first_file = 'default', area = 0.5):
    '''
    Plot the Hybdrt taken near OCV for each pol-map taken during a stability test
    Only would have been used for cells 113-118

    Parameters:
    -----------
    data_path, str:
        path to where the pol map datafiles are stored
    first_file, str" (default = 'default')
        finds the first ocv file to attain the time the test started
        if set to 'default' the init ocp file is used as the beginning time
        if there is a specific file that should be first, set first_file to the path to that file
    area, float: (default = 0.5)
        active area of the cell

    Return --> None, but a plot of all the DRT spectra take at OCV throughout the test are plotted
    Also the ohmic, rp, and DRT peak data is saved to an excel file
    '''
    # --- Attaining relavent files for hybdrt
    ocv_tests = find_files_with_string(data_path, 'Staircase-charge_0')
    eis_files = [s for s in ocv_tests if 'EISGALV' in s]
    chrono_files = [s for s in ocv_tests if '0-a_Filtered' in s]

    # --- Sorting the files in the order that they were taken
    # - Define a function to extract the numeric value from the identifier
    def extract_number(file_name):
        if 'PolMap-Init-' in file_name:
            return -1  # Special case: 'PolMap-Init-' should come first
        match = re.search(r'PolMap-(\d+)-', file_name)
        return int(match.group(1)) if match else float('inf')

    # - Sort the files based on the numeric value
    sorted_eis_files = sorted(eis_files, key=extract_number)
    sorted_chrono_files = sorted(chrono_files, key=extract_number)

    # --- Attaining the beginning time of the test
    if first_file == 'default':
        t0_stamp = init_ocp_timestamp(data_path,'FC_Stb_PolMap')
        t0 = int(t0_stamp.strftime("%s"))
    else:
        t0_stamp = fl.get_timestamp(first_file)
        t0 = int(t0_stamp.strftime("%s"))

    # --- Attaining the time each DRT was taken
    test_times = []
    for file in sorted_chrono_files:
        time = fl.get_timestamp(file) # datetime object
        time_int = int(time.strftime("%s")) # s (integer)
        delta_t_hours = int((time_int-t0)/3600) # hrs (integer)
        test_times.append(delta_t_hours)

    # ---- Plotting -- Need to figure out how to plot just the DRT and all on one plot
    fig, ax = plt.subplots()
    end_time = test_times[-1]

    # - Setting the colorscheme
    cmap = cmr.get_sub_cmap('cmr.rainforest', 0.10, 0.90)
    color_space = np.linspace(0,1,end_time) # array from 0-1 for the colormap for plotting

    # - Initializing Lists:
    ohmic_asr = np.array([]) # ohm*cm^2
    rp_asr = np.array([]) # ohm*cm^2
    df_tau_r = pd.DataFrame(columns = ['Time (hrs)','Tau','Resistance']) #Initializing DataFrame to save DRT peak resistances

    for eis, chrono, time in zip(sorted_eis_files, sorted_chrono_files, test_times):
        color = cmap(color_space[time-1])
        # plot_hybdrts(chrono,eis,ax=ax,color = color) # Plot DRT

        # --- Fitting
        hy_tup = fl.get_hybrid_tuple(chrono, eis)
        drt = DRT() # Create a DRT instance
        drt.fit_hybrid(*hy_tup, subtract_background=False, background_type='dynamic',
                background_corr_power=0.5)

        # --- Plotting
        drt.plot_distribution(area=area,ax=ax,color = color)

        # - Formatting
        
        # - Excessive formatting
        ax.tick_params(axis='both', which='major', labelsize='x-large') #changing tick label size
        ax.xaxis.get_label().set_fontsize('xx-large')
        ax.yaxis.get_label().set_fontsize('xx-large')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Colorbar formatting
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([0,end_time])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
        cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
        # cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
        cb.ax.tick_params(labelsize='x-large')

        # --- Saving Data to lists
        ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area) # Ω * cm^2
        rp_asr = np.append(rp_asr,drt.predict_r_p() * area) # Ω * cm^2
        append_hybdrt_peaks(df_tau_r, drt, area, time)
    
    plt.tight_layout()
    plt.show()

    # --- Appending Data to excel:
    folder_loc = os.path.dirname(data_path)
    cell_name = find_cell_name(folder_loc)

    rp_ohmic_to_excel(cell_name, folder_loc, ohmic_asr, rp_asr, 'PolMap_fc_stb_hybdrt', test_times, 'Time (Hrs)', overwrite = True)
    df_tau_r_to_excel(cell_name, folder_loc, df_tau_r,'PolMap_fc_stb_hybdrt_peaks', overwrite = True)

def plot_polmap_near_ocv_stb_r(data_path:str):
    '''
    Plots Rp and ohmic over time and peak resistance overtime
    for the HybDRT OCV data from a pol-map stability test
    The data must already be fit and appended to the excel data file
    OCV data is the first charge file (which is slightly EC)
    Only would have been used for cells 113-118


    Parameters:
    -----------
    data_path, str:
        path to where the pol map datafiles are stored

    Return --> None, two subplots are plot and shown as one figure
    '''
    # - Setup
    folder_loc = os.path.dirname(data_path) # Getting location of main data folder
    fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8)) # Initialize figure
    cmap = cmr.get_sub_cmap('cmr.rainforest', 0.10, 0.90) # Initialize colormap

    # -- Plotting the cell resistance over time
    try:
        plot_r_over_time(folder_loc, 'PolMap_fc_stb_hybdrt', ax = axs[0], cmap = cmap)
    except ValueError as e:
        print(f"ValueError: {e}")
        print('This error is triggered because there is no resistance data found')
        print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
        print('Run the plot_stb_ocv_drt function to analyze and append the data')

    try:
        plot_drt_peaks(folder_loc, 'PolMap_fc_stb_hybdrt_peaks', 'Time (hrs)', ax = axs[1], cmap = cmap,
                        test_type = 'stb')
    except ValueError as e:
        print(f"ValueError: {e}")
        print('This error is triggered because there is no drt peak data found')
        print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
        print('Run the plot_stb_ocv_drt function to analyze and append the data')

    plt.show()

def plot_stb_eisdrt(data_path, first_file = 'default', area = 0.5, peaks_to_fit = 'best_id'):
    '''
    Plot EIS data taken at OCV during a pol-map stability test

    Parameters:
    -----------
    data_path, str:
        path to where the pol map datafiles are stored
    first_file, str" (default = 'default')
        finds the first ocv file to attain the time the test started
        if set to 'default' the init ocp file is used as the beginning time
        if there is a specific file that should be first, set first_file to the path to that file
    area, float: (default = 0.5)
        active area of the cell
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amount of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer

    Return --> None, but a plot of all the DRT spectra take at OCV throughout the test are plotted
    Also the ohmic, rp, and DRT peak data is saved to an excel file
    '''
    # --- Attaining relavent files for hybdrt
    eis_files = find_files_with_string(data_path, 'EISPOT')

    # --- Sorting the files in the order that they were taken
    # - Define a function to extract the numeric value from the identifier
    def extract_number(file_name):
        if 'PolMap-Init-' in file_name:
            return -1  # Special case: 'PolMap-Init-' should come first
        match = re.search(r'PolMap-(\d+)-', file_name)
        return int(match.group(1)) if match else float('inf')

    # - Sort the files based on the numeric value
    sorted_eis_files = sorted(eis_files, key=extract_number)
    sorted_eis_files = [sorted_eis_files[-1]] + sorted_eis_files[:-1]

    # --- Attaining the beginning time of the test
    if first_file == 'default':
        t0_stamp = init_ocp_timestamp(data_path,'FC_Stb_PolMap')
        t0 = int(t0_stamp.strftime("%s"))
    else:
        t0_stamp = fl.get_timestamp(first_file)
        t0 = int(t0_stamp.strftime("%s"))

    # --- Attaining the time each DRT was taken
    test_times = []
    for file in sorted_eis_files:
        time = fl.get_timestamp(file) # datetime object
        time_int = int(time.strftime("%s")) # s (integer)
        delta_t_hours = int((time_int-t0)/3600) # hrs (integer)
        test_times.append(delta_t_hours)

    ' =_=_=_=_=_= Plotting '
    # --- Plotting EIS
    # - Setting the colorscheme
    end_time = test_times[-1]
    cmap = cmr.get_sub_cmap('cmr.rainforest', 0.10, 0.90)
    color_space = np.linspace(0,1,end_time) # array from 0-1 for the colormap for plotting

    # - Plotting
    fig,ax = plt.subplots()
    for test, peis in enumerate(sorted_eis_files):
        # - Finding time of the EIS from the start of the degradation test
        time = test_times[test]
        nyquist_name =  str(time) + ' Hours'
        if time == 0:
            color = cmap(color_space[time])
        else:
            color = cmap(color_space[time-1])
        
        # - Plotting
        df_useful = get_eis_data(peis)
        df_useful['ohm'] = df_useful['ohm'] * area
        df_useful['ohm.1'] = df_useful['ohm.1'] * -area
    
        ax.plot(df_useful['ohm'],df_useful['ohm.1'],'o',markersize=9,label = nyquist_name,color = color) #plots data

        # - Plot Formatting                  
        ax.set_xlabel('Zreal (\u03A9 cm$^2$)',fontsize='xx-large')
        ax.set_ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',fontsize='xx-large')
        ax.tick_params(axis='both', which='major', labelsize='x-large')
        ax.axhline(y=0,color='k', linestyle='-.') # plots line at 0 #D2492A is the color of Mines orange
        ax.axis('scaled') #Keeps X and Y axis scaled 1 to 1
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # - Color bar formatting
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([0,end_time])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
    cb.set_label(label='Time (hrs)',size='x-large',labelpad = -10)
    # cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
    cb.ax.tick_params(labelsize='x-large')

    plt.tight_layout()
    plt.show()

    # --- Plotting DRT:
    fig, ax = plt.subplots()

    # - Initializing Lists:
    ohmic_asr = np.array([]) # ohm*cm^2
    rp_asr = np.array([]) # ohm*cm^2
    df_tau_r = pd.DataFrame(columns = ['Time (hrs)','Tau','Resistance']) #Initializing DataFrame to save DRT peak resistances

    for eis, time in zip(sorted_eis_files, test_times):
        if time == 0:
            color = cmap(color_space[time])
        else:
            color = cmap(color_space[time-1])
        label = str(time) + ' Hours'

        # --- Fitting and plotting
        drt = quick_dualdrt_plot(eis, area, label=label, ax=ax, peaks_to_fit = peaks_to_fit,
                            mark_peaks=False, scale_prefix = "", legend = False, color=color) 
        
        # - Excessive formatting
        ax.tick_params(axis='both', which='major', labelsize='x-large') #changing tick label size
        ax.xaxis.get_label().set_fontsize('xx-large')
        ax.yaxis.get_label().set_fontsize('xx-large')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Colorbar formatting
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([0,end_time])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
        cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
        # cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
        cb.ax.tick_params(labelsize='x-large')

        # --- Saving Data to lists
        ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area) # Ω * cm^2
        rp_asr = np.append(rp_asr,drt.predict_r_p() * area) # Ω * cm^2
        append_hybdrt_peaks(df_tau_r, drt, area, time)
    
    plt.tight_layout()
    plt.show()

    # --- Appending Data to excel:
    folder_loc = os.path.dirname(data_path)
    cell_name = find_cell_name(folder_loc)

    rp_ohmic_to_excel(cell_name, folder_loc, ohmic_asr, rp_asr, 'PM_fc_stb_eis', test_times, 'Time (Hrs)', overwrite = True)
    df_tau_r_to_excel(cell_name, folder_loc, df_tau_r,'PM_fc_stb_eis_peaks', overwrite = True)

def plot_stb_r_overtime(data_path:str):
    '''
    Plots Rp and ohmic over time and peak resistance overtime
    for the HybDRT OCV data from a pol-map stability test
    The data must already be fit and appended to the excel data file

    Parameters:
    -----------
    data_path, str:
        path to where the pol map datafiles are stored

    Return --> None, two subplots are plot and shown as one figure
    '''
    # - Setup
    folder_loc = os.path.dirname(data_path) # Getting location of main data folder
    fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8)) # Initialize figure
    cmap = cmr.get_sub_cmap('cmr.rainforest', 0.10, 0.90) # Initialize colormap

    # -- Plotting the cell resistance over time
    try:
        plot_r_over_time(folder_loc, 'PM_fc_stb_eis', ax = axs[0], cmap = cmap)
    except ValueError as e:
        print(f"ValueError: {e}")
        print('This error is triggered because there is no resistance data found')
        print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
        print('Run the plot_stb_ocv_drt function to analyze and append the data')

    try:
        plot_drt_peaks(folder_loc, 'PM_fc_stb_eis_peaks', 'Time (hrs)', ax = axs[1], cmap = cmap,
                        test_type = 'stb')
    except ValueError as e:
        print(f"ValueError: {e}")
        print('This error is triggered because there is no drt peak data found')
        print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
        print('Run the plot_stb_ocv_drt function to analyze and append the data')

    plt.show()

' ---- IV plotting functions'
def plot_iv(loc, area, CD_at_V = 1.3):
    '''
    '''
    df = get_iv_data(loc) # Reads the .DTA file and saves the IVFC data as a dataframe

    # -calculations and only keeping the useful data
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)

    df_useful = df[['V vs. Ref.','A','W']]
    df_useful = df_useful.rename(columns={'V vs. Ref.': 'V'})

    # - Creating a new df of just the fc mode data
    jump_threshold = 16
    index_of_jump = (df['s'].diff() > jump_threshold).idxmax()
    df_fc = df.loc[:index_of_jump-1]
    df_ec = df.loc[index_of_jump:]

    ' ---------- Plotting ----------'
    fig, ax1 = plt.subplots() # Initialize figure
    # ax1.plot(df_useful['A'], df_useful['V'],'o')
    # - Colors
    c_fc ='#ff7f00' #'#AB9B1F' '#ff7f00' '#D36135'  '#21314D' # Mines Navy
    c_ec = '#007bff' #'#263AD4' '#007bff' '#7FB069' '#93a1ba' # Mines Light blue
    # c_fcppd = '#1D8A99' # '#D2492A' # Mines orange color

    # - Current Density Plot
    ax1.plot(df_fc['A'], df_fc['V vs. Ref.'],'o',color=c_fc)
    ax1.plot(df_ec['A'], df_ec['V vs. Ref.'],'o', color=c_ec)

    # --- Formatting
    ax1.set_xlabel('Current Density ($A/cm^2$)',fontsize = 'xx-large')
    ax1.set_ylabel('Voltage (V)', fontsize = 'xx-large')
    ax1.axvline(x=0, color = 'k', linewidth=2, linestyle='--')

    # - Power density plotting
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Power Density ($W/cm^2$)', fontsize = 'xx-large', labelpad=-10)  # we already handled the x-label with ax1
    ax2.plot(df_fc['A'], df_fc['W'], 'o',color=c_fc,mfc='none') 
    ax2.tick_params(axis='y', labelsize = 'x-large')

    # --- Calculating and printing max values onto the graph
    # - FC
    max_w = df_fc['W'].max() #finds maximum power density
    # max_ws = f'{round(max_w,3)}' #sets float to a string
    # plt.figtext(0.49,0.18,r'$P_{max} = $'+ max_ws + r' $W/cm^2$',size='x-large',weight='bold',color=c_fc)

    # - EC
    CD_at_mod = CD_at_V-0.01
    current_density15 = df_useful[abs(df_useful['V'])>=CD_at_mod].iloc[0,1] # Finds the current density of the first Voltage value above the desired Voltage
    V15 = df_useful[abs(df_useful['V'])>=CD_at_mod].iloc[0,0] # Same as before but returns the exact voltage value
    current_density15_string = f'{round(current_density15,3)}'
    V15_string = f'{round(V15,3)}'
    # plt.figtext(0.13,0.18,current_density15_string+r' $A/cm^2\:at$ '+V15_string+r'$V$',size='x-large',weight='bold', c=c_ec) #placing value on graph

    # - Excessive formatting;
    ax1.tick_params(axis='y', labelsize = 'x-large')
    ax1.tick_params(axis='x',labelsize = 'x-large')
    plt.figtext(0.30,0.91,'EC', fontsize = 'xx-large', color = c_ec)
    plt.figtext(0.60,0.91,'FC', fontsize = 'xx-large', color = c_fc)

    # Adjusting y axis for ppd
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    yticks_ppd = ax2.get_yticks()
    yticks_ppd = yticks_ppd[1:-2]
    yticks_ppd = np.append(yticks_ppd, round(max_w,3))
    ax2.set_yticks(yticks_ppd)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    last_tick_label = ax2.get_yticklabels()[-1]
    last_tick_line = ax2.get_yticklines()[-1]
    last_tick_label.set_color(c_fc)
    last_tick_line.set_markeredgecolor(c_fc)
    last_tick_label.set_weight('bold')  # Make the label bold

    # adjusting x axis for CD in electrolysis cell mode
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    minor_locator = ticker.FixedLocator([round(current_density15,3)])  # Adjust the value to the desired position
    ax1.xaxis.set_minor_locator(minor_locator)
    ax1.tick_params(axis='x', which='minor',direction='in',length=4,color=c_ec,width=1,pad=-15,
                    labelsize='x-large',labelcolor=c_ec)
    ax1.set_xticklabels([current_density15_string],minor=True,weight='bold')
    ax1.xaxis.set_minor_formatter(FormatStrFormatter("%.3g"))

    # Adjusting y axis for the Current Density at a certian voltage
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ticks = ax1.get_yticks()
    ticks = ticks[1:-1]
    ticks = np.append(ticks, round(V15,3))
    ax1.set_yticks(ticks)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
    last_tick_label = ax1.get_yticklabels()[-1]
    last_tick_line = ax1.get_yticklines()[-1]
    last_tick_label.set_color(c_ec)
    last_tick_line.set_markeredgecolor(c_ec)
    last_tick_label.set_weight('bold')  # Make the label bold

    # Adjusting Voltage axis to have the OCV value
    ocv = round(float(get_iv_ocv(loc)),3)
    def find_closest_float_index(arr, target):
        return min(range(len(arr)), key=lambda i: abs(arr[i] - target))
    v_ticks = ax1.get_yticks()

    closest_index = find_closest_float_index(v_ticks, ocv)
    v_ticks[closest_index] = ocv
    ax1.set_yticks(ticks)
    ocv_tick_label = ax1.get_yticklabels()[closest_index]
    ocv_tick_label.set_weight('bold')


    ax2.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)


    plt.tight_layout()
    plt.show()


' -------- Helper Functions ---------- '
def find_ocp_files(directory):
    '''
    Finds all the ocp files
    95% written by Chat GPT
    
    '''
    ocp_files = []
    
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the filename contains 'ocp'
            if 'ocp' in file.lower() and identifier in file:
                # If yes, add the full path to the list
                ocp_files.append(os.path.join(root, file))

    return ocp_files

def init_ocp_timestamp(directory,pattern):
    '''
    Gets the initial timestamp
    95% written by ChatGPT
    
    '''
    earliest_timestamp = datetime.max # Initialize with positive infinity

    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'ocp' in file.lower() and pattern in file:
                file_path = os.path.join(root, file)
                timestamp = fl.get_timestamp(file_path)
                earliest_timestamp = min(earliest_timestamp, timestamp)

    return earliest_timestamp

def find_files_with_string(directory, target_string):
    '''
    Finding all files with a certain string
    Written by Chat GPT

    Parameters
    '''
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if target_string in file:
                file_path = os.path.join(root, file)
                matching_files.append(file_path)

    return matching_files

def concatenate_data(file_list:str, data_type:str):
    '''
    Concatenates many different files containing potentiostatic or OCV data

    Parameters:
    -----------
    file_list, str:
        list of datafiles to concatinate
    data_type, str:
        Type of data attaining. For now the two options are 'pot' and 'ocv'
        Use 'pot' to attain potentiostatic hold data
        Use 'ocv' to attain OCV data
    
    Return --> pandas dataframe containing all the desired pstat or ocv data
    '''
    dfs = [] #Initializing list of dfs

    # - Iterate through all the files and make a list of dataframes containing the useful data
    for file in file_list:
        df = get_ocv_data(file)
        start_time = fl.get_timestamp(file).strftime("%s") # Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        if data_type == 'pot':
            df_useful = df[['s','A']]
        elif data_type == 'ocv':
            df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)

    # - Combining all of the DataFrames and converting time to hours
    cat_dfs = pd.concat(dfs,ignore_index=True) # (s) Combine all the DataFrames in the file folder

    return cat_dfs

def append_hybdrt_peaks(df_tau_r, drt, area, condition):
    '''
    Takes a hybDRT instance and extracts the peak data from it. It then appends this data to a dataframe (df_tau_r)
    The dataframe is made outside the function
    
    Parameters:
    ----------
    df_tau_r, pandas dataframe:
        dataframe to append the drt peak tau and resistance data to
    drt, drt instance:
        DRT instance
    area, float:
        The active cell area in cm^2
    condition, float:
        the condition of interest that the drt data was taken in
    
    Return --> None
    '''
    # --- obtain time constants from inverters
    peak_tau_list = drt.find_peaks() # - τ/s
    r_list = drt.predict_distribution(peak_tau_list)

    # - Normalizing peak resistance to active cell area
    r = np.array(r_list) * area # Ω*cm2

    for i,τ in enumerate(peak_tau_list):
        df_tau_r.loc[len(df_tau_r.index)] = [condition, τ, r[i]]
        i = i+1