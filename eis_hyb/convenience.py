''' This module contains functions to format, plot, analyze, and save EIS and DRT spectra for multiple 
Electrochemical cell tests.  The data comes from a Gamry Potentiostat
# C-Meisel
'''
# import bayes_drt2
'Imports'
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
import natsort
import seaborn as sns
import scipy as scipy
import cmasher as cmr
import sys # Used to go through the system
import traceback # Used to look at errors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from hybdrt.models import DRT, elements, drtbase
import hybdrt.plotting as hplt
from hybdrt.fileload import read_eis, get_eis_tuple

from .plotting import plot_peis, plot_peiss, lnpo2
from .fit_drt import dual_drt_save, pfrt_drt_save
from .data_formatting import peis_data


def standard_performance(loc:str, jar:str, area:float=0.5, **peis_args):
    '''
    Plots the EIS, dual-fits and plots the DRT, and prints the ohmic and polarization resistance for a cell
    Generally this function is used on the first EIS spectra taken at standard testing conditions

    Parameters
    ----------
    loc, str: (path to a directory)
        The location of the folder containing the EIS files (path to folder)
    jar, str: (path to a directory)
        The location of the jar containing the map-fits (path to jar)
    area, float: 
        The active cell area in cm^2 (default = 0.5)
    peis_args, dict: 
        Any additional arguments to be passed to the plot_peis function

    Return --> None, but it plots EIS and DRT. It also prints the ohmic and rp values of the cell
    '''
    cell_name = os.path.basename(loc).split("_", 1)[0] #gets the name of the cell

    # --- Fitting DRT, and making sure I am not re-fitting it if it has already been fit
    pickle_jar = os.listdir(jar)

    # # - Dual fit check - Will place back in later
    # dual_pickle_name = 0
    # dual_fit_file = cell_name + '_standard_dual.pkl'
    # for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
    #     if dual_fit_file == pickle:
    #         dual_pickle_name = dual_pickle_name + 1
    #         break

    # if dual_pickle_name == 0:
    #     dual_fit_name = dual_fit_file.removesuffix('_dual.pkl')
    #     dual_drt_save(loc,jar,dual_fit_name)

    # - Dual fit
    dual_drt = DRT() # Create a DRT instance
    df = read_eis(loc)
    freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe
    dual_drt.dual_fit_eis(freq, z, discrete_kw=dict(prior=True, prior_strength=None)) # Fit the data

    best_id = dual_drt.get_best_candidate_id('discrete', criterion='lml-bic')

    # - Probability function of relaxation time fit check
    pfrt_pickle_name = 0
    pfrt_fit_file = cell_name+'_standard_pfrt.pkl'
    for pickle in pickle_jar: # checks to see if this has already been fit, if so name gets set to 1
        if pfrt_fit_file == pickle:
            pfrt_pickle_name = pfrt_pickle_name + 1
            break

    if pfrt_pickle_name == 0:
        pfrt_fit_name = pfrt_fit_file.removesuffix('_pfrt.pkl')
        pfrt_drt_save(loc,jar,pfrt_fit_name)

    # ---- Loading DRT and plotting
    # dual_drt = DRT()
    pfrt_drt = DRT()

    # dual_drt.load_attributes(os.path.join(jar,dual_fit_file))
    pfrt_drt.load_attributes(os.path.join(jar,pfrt_fit_file))

    pfrt_basis_tau = pfrt_drt.basis_tau # gaining the basis tau of the pfrt fit
    pf = pfrt_drt.predict_pfrt(pfrt_basis_tau) # saving the pfrt as an object
    
    # -- Plotting
    fig, ax = plt.subplots()
    m_blue = '#21314D'
    m_orange = '#c1741d'
    
    # dual_drt.plot_distribution(c=m_blue, plot_ci=True, label='Dual DRT', return_line=True, ax=ax, area=area, mark_peaks = True)
    dual_drt.plot_candidate_distribution(best_id, 'discrete', label='Dual DRT',
                c=m_blue, ax=ax, area=area)

    ax2 = ax.twinx()
    ax2.plot(pfrt_basis_tau, pf, label='PFRT', color=m_orange)
    
    # - Formating
    ax2.set_ylabel('$p$',fontsize='xx-large',color=m_orange, labelpad = -10)
    ax.legend(loc='upper left', fontsize = 'x-large', frameon = False, labelcolor = m_blue, handletextpad=0.3)
    ax2.legend(fontsize = 'x-large', frameon = False, labelcolor = m_orange, handletextpad=0.3)

    # - Excessive formatting
    ax.xaxis.label.set_size('xx-large')
    ax.tick_params(axis='x', labelsize='x-large')

    ax.yaxis.label.set_size('xx-large')
    ax.yaxis.label.set_color(m_blue)
    ax.tick_params(axis='y', labelsize='x-large', labelcolor=m_blue, color=m_blue)
    ax.spines['left'].set_color(m_blue) 

    ax2.tick_params(axis='y', labelsize='x-large', labelcolor=m_orange, color=m_orange)
    ax2.spines['right'].set_color(m_orange)
    ax2.set_yticks([0,1])

    ax2.spines['top'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    plt.show()

    # # ---- Plotting EIS
    ohmic = dual_drt.predict_r_inf()*area
    rp = dual_drt.predict_r_p()*area
    ohmic_rtot = [round(float(ohmic),2),round(float(rp),2)+round(float(ohmic),2)]
    plot_peis(area,loc,**peis_args) #Plots standard PEIS spectra

    print('Standard Ohmic',round(ohmic,3),'\u03A9cm\u00b2')
    print('Standard Rp',round(rp,3),'\u03A9cm\u00b2')

def quick_dualdrt_plot(loc:str, area:float, label:str = None, ax:plt.Axes = None, peaks_to_fit:int = 'best_id'):
    '''
    Quicker version to fit and plot a dual DRT spectra to EIS data.
    This function supports multiple graphs stacked on each other
    If one spectra is being plot, the fig and ax are made inside the function.

    Parameters
    ----------
    loc, str: (path to a directory)
        The location of the folder containing the EIS files (path to folder)
    area, float:
        The active cell area in cm^2
    ax, plt.Axes:
        matplotlib axes object. Used to plot the spectra. Needed to plot different spectra on the same figure
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amout of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer
    
    Return --> None, but it plots and shows one or more plots
    '''
    # -- Gathering data
    drt = DRT()
    df = read_eis(loc)
    freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe
    drt.dual_fit_eis(freq, z, discrete_kw=dict(prior=True, prior_strength=None)) # Fit the data
    tau = drt.get_tau_eval(20)


    # -- Selecting the number of peaks for the drt distribution to have.
    if peaks_to_fit == 'best_id':
        best_id = drt.get_best_candidate_id('discrete', criterion='lml-bic')
        peaks = best_id

    else: peaks = peaks_to_fit

    # - Selecting the model. The number of peaks to plot and fit
    model_dict = drt.get_candidate(peaks,'discrete')
    model = model_dict['model']
    
    # --- Plotting
    solo = None 
    if ax == None: # If only one spectra is being plot, create the fig and axis in the function
        solo = True
        fig, ax = plt.subplots()


    model.plot_distribution(tau, ax=ax, area=area,label=label,mark_peaks=True)

    # - Formatting
    ax.legend(fontsize='x-large')
    ax.xaxis.label.set_size('xx-large') 
    ax.yaxis.label.set_size('xx-large')
    ax.tick_params(axis='both', labelsize='x-large')

    # - Excessive formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if solo == True:
        plt.tight_layout()
        plt.show()

def po2_plots_dual(folder_loc:str, fit_path:str, area:float, eis:bool=True, drt:bool=True,
                 o2_dependence:bool=True, drt_peaks:bool=True, print_resistance_values:bool=False,
                  ncol:int=1, legend_loc:str='best', flow100:bool = False, cut_inductance:bool = False,
                  overwrite:bool = False, peaks_to_fit:int = 'best_id'):
    '''
    Searches through the folder_loc for all the changes in O2 concentration EIS files.
    Plots the EIS for each concentration in one plot if eis=True and the DRT of each concentration in another if drt=True
    dual-fits all of the eis files.
    If O2_dependence = True, plots the ln(1/ASR) vs. ln(O2 concentration) for the ohmic and polarization resistance
    This funciton uses dual_drt regression fitting.

    Parameters:
    -----------
    folder_loc, str: (path to folder)
        The location of the folder containing the EIS files 
    fit_path, str: (path to folder)
        The location of the folder to save the map-fits
    area, float:
        The active cell area in cm^2
    eis, bool: (default = True)
        If True, plots the EIS for each concentration in one plot
    drt, bool: (default = True)
        If True, plots the DRT for each concentration in another plot
    o2_dependence, bool: (default = True)
        If True, plots the ln(1/ASR) vs. ln(O2 concentration) for the ohmic and polarization resistance
    drt_peaks, bool: (default = True)
        If True, the DRT peaks are fit and plotted. If the peaks have not been fit, they
        will be fit and added the the cell data excel sheet. If the data exists in the spreadsheet already
        then that data will be plotted.
    print_resistance_values, bool: (default = False)
        If True, prints the ohmic and polarization resistance for each concentration
    ncol, int: (default = 1)
        The number of columns to use for the plot legend 
    legend_loc, str: 
        The location of the legend (default = 'best'). The other option is to put 
        the legend outside the figure and this is done by setting legend_loc = 'outside'
    flow100, bool: (default = False)
        Set this to true if the total flow rate for the PO2 test was 100 SCCM
        for my older tests (cell 16 and below) my total flow rate was 100 SCCM
        This just changes the list of strings to look for
    cut_inductance, bool: (default = False)
        If this is set to true, the negative inductance values at the beginning of the DataFrame
    overwrite, bool: (default = False)
        If the data already exists in the cell excel file, then this will overwrite that data with
        the data currently being fit when overwrite=True.
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amout of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer

    Return --> None, but it plots and shows one or more plots
    '''
    
    # +++++++++----- Finding and Dual DRT fitting the PO2 EIS
    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc
    substring_list = ['PEIS_20O2.80Ar','PEIS_40O2.60Ar','PEIS_60O2.40Ar','PEIS_80O2.20Ar','PEIS_100O2.0Ar'] # setting variables to look for
    O2_conc_list = [dta_files for dta_files in dta_files if any(sub in dta_files for sub in substring_list)] # placing all the changes in O2 eis files into a list
    cell_name = os.path.basename(fit_path).split("_", 2)[1]

    O2_conc_list = sorted(O2_conc_list, key=lambda x: int((x[x.find('PEIS_')+len('PEIS_'):x.rfind('O2')]))) #Sorts numerically by PO2
    
    ' ---------- Plotting all the PO2 concentration EIS spectra'
    if eis == True:
        plt.figure() # Initializing the Figure

        for peis in O2_conc_list:
            # --- Finding and formatting
            loc = os.path.join(folder_loc, peis) # creates the full path to the file
            po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name

            nyquist_name =  po2 + '% O$_2$'

            if flow100 == True:
                po2_int = int(po2)
                po2 = str(po2_int * 2)
                nyquist_name = po2 + '% O$_2$'

            # --- Plotting
            plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc=legend_loc,cut_inductance = cut_inductance)

        plt.show()

    ' ---------- inverting, plotting, and dual fitting all the PO2 concentration DRT spectra'
    if drt == True: 
        fig, ax = plt.subplots() #initializing plots for DRT
        # --- Initializing lists for further analysis
        O2_conc = np.array([]) # Concentration of Oxygen
        ohmic_asr = np.array([]) # ohm*cm^2
        rp_asr = np.array([]) # ohm*cm^2
        df_tau_r = pd.DataFrame(columns = ['O2 Concentration (%)','Tau','Resistance']) #Initializing DataFrame to save temperature

        for peis in O2_conc_list: #For loop to plot the DRT of all PO2 files (probably a more elegant way, but this works)
            # --- Finding and formatting
            loc = os.path.join(folder_loc, peis) # creates the full path to the file
            po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name

            nyquist_name =  po2 + '% O$_2$'

            if flow100 == True:
                po2_int = int(po2)
                po2 = str(po2_int * 2)
                nyquist_name = po2 + '% O$_2$'

            # --- Inverting the EIS data
            drt = DRT()
            df = read_eis(loc)
            freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe
            drt.dual_fit_eis(freq, z, discrete_kw=dict(prior=True, prior_strength=None)) # Fit the data
            label = po2+ '% O$_2$'

            # drt.plot_distribution(mark_peaks=True, label=label, ax=ax, area=area)
            tau = drt.get_tau_eval(20)

            # - Selecting the number of peaks for the drt distribution to have.
            if peaks_to_fit == 'best_id':
                best_id = drt.get_best_candidate_id('discrete', criterion='lml-bic')
                peaks = best_id

            else: peaks = peaks_to_fit

            # - Selecting the model. The number of peaks to plot and fit
            model_dict = drt.get_candidate(peaks,'discrete')
            model = model_dict['model']

            
            # --- Plotting
            label = po2+ '% O$_2$'
            # drt.plot_candidate_distribution(peaks, 'discrete',mark_peaks=False, label=label, ax=ax, area=area)
            model.plot_distribution(tau, ax=ax, area=area,label=label,mark_peaks=True)

            # --- Appending resistance values to lists for lnPO2 plotting
            O2_conc = np.append(O2_conc,float(po2)/100)
            ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area)
            rp_asr = np.append(rp_asr,drt.predict_r_p() * area)

            # --- obtain time constants from inverters and Appending tau and r for each peak into df_tau_r
            tau = model_dict['peak_tau'] # τ/s

            # - Obtaining the resistance value for each peak
            r_list = []
            for i in range(1,int(peaks)+1):
                peak = 'R_HN'+str(i)
                resistance = model.parameter_dict[peak]
                r_list.append(resistance)

            r = np.array(r_list) * area # Ω*cm2

            i = 0
            for τ in tau:
                df_tau_r.loc[len(df_tau_r.index)] = [po2, τ, r[i]]
                i = i+1

            # --- Printing the resistance values if that is desired
            if print_resistance_values == True:
                print(po2 + '% Oxygen Ohmic', round(drt.predict_r_inf() * area,3), '\u03A9 cm$^2^')
                print(po2 + '% Oxygen Rp', round(drt.predict_r_p() * area,3), '\u03A9 cm$^2^')
        
        # - Formatting
        # print(df_tau_r)
        ax.legend(fontsize='x-large')
        ax.xaxis.label.set_size('xx-large') 
        ax.yaxis.label.set_size('xx-large')
        ax.tick_params(axis='both', labelsize='x-large')
        
        plt.tight_layout()
        plt.show()

        # >>>>>>>>>>>> Creating DataFrames and adding excel data sheets (or creating the file if need be)'
        # ------- Cell Resistance Data
        excel_name = '_' + cell_name + '_Data.xlsx'
        excel_file = os.path.join(folder_loc,excel_name)
        sheet_name = 'pO2_dual'
        exists = False

        exists, writer = excel_datasheet_exists(excel_file,sheet_name)

        if exists == False:
            df_po2 = pd.DataFrame(list(zip(O2_conc*100,ohmic_asr,rp_asr)),
                columns =['O2 Concentration (%)','Ohmic ASR (ohm*cm$^2$)', 'Rp ASR (ohm*cm$^2$)'])
            df_po2.to_excel(writer, sheet_name=sheet_name, index=False) # Writes this DataFrame to a specific worksheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

        elif exists == True and overwrite == True:
            df_po2 = pd.DataFrame(list(zip(O2_conc*100,ohmic_asr,rp_asr)),
                columns =['O2 Concentration (%)','Ohmic ASR (ohm*cm$^2$)', 'Rp ASR (ohm*cm$^2$)'])
            
            book = load_workbook(excel_file)
            writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_po2.to_excel(writer, sheet_name=sheet_name, index=False) # Extract data to an excel sheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

            df_po2 = pd.read_excel(excel_file,sheet_name)

        # ------- Appending the Peak_fit data to an excel file
        excel_name = '_' + cell_name + '_Data.xlsx'
        excel_file = os.path.join(folder_loc,excel_name)
        peak_data_sheet = 'pO2_dual_DRT_peaks'
        exists_peaks = False

        exists_peaks, writer_peaks = excel_datasheet_exists(excel_file,peak_data_sheet)
    
        if exists_peaks == False: # Make the excel data list
            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.
        
        elif exists_peaks == True and overwrite == False: #load the data into a DataFrame
            df_tau_r = pd.read_excel(excel_file,peak_data_sheet)

        elif exists_peaks == True and overwrite == True:
            book = load_workbook(excel_file)
            writer_peaks = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.

            df_tau_r = pd.read_excel(excel_file,peak_data_sheet)


    ' ------ Plotting the Oxygen dependence'
    if o2_dependence == True:
        lnpo2(ohmic_asr,rp_asr,O2_conc)
    elif drt==False and o2_dependence==True or print_resistance_values==True:
        print('Set drt to True. The cell resistance values are found using Jakes DRT package and thus the EIS spectra need to be fit')

    ' --- DRT peak plotting --- '
    if drt_peaks == True:
        # ----- plotting
        palette = sns.color_palette('crest_r', as_cmap=True)
        plot = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df_tau_r, hue='O2 Concentration (%)',palette = palette,s=69)

        # ----- Ascetics stuff
        sns.set_context("talk")
        fontsize = 14
        sns.despine()
        plot.set_ylabel('ASR (\u03A9 cm$^2$)',fontsize=fontsize)
        plot.set_xlabel('Time Constant (\u03C4/s)',fontsize=fontsize)
        plot.set(xscale='log')

        plt.tight_layout()
        plt.show()

def po2_plots_save(folder_loc:str, fit_path:str, area:float, eis:bool=True, drt:bool=True,
                 o2_dependence:bool=True, drt_peaks:bool=True, print_resistance_values:bool=False,
                  ncol:int=1, legend_loc:str='best', flow100:bool = False, cut_inductance:bool = False,
                  overwrite = False):
    '''
    Searches through the folder_loc for all the changes in O2 concentration EIS files.
    PFRT-fits all of hte eis files and saves them to fit_path. If the files are already fit, they will not be re-fit.
    Plots the EIS for each concentration in one plot if eis=True and the DRT of each concentration in another if drt=True
    If O2_dependence = True, plots the ln(1/ASR) vs. ln(O2 concentration) for the ohmic and polarization resistance

    Parameters:
    -----------
    folder_loc, str: (path to folder)
        The location of the folder containing the EIS files 
    fit_path, str: (path to folder)
        The location of the folder to save the map-fits
    area, float:
        The active cell area in cm^2
    eis, bool: (default = True)
        If True, plots the EIS for each concentration in one plot
    drt, bool: (default = True)
        If True, plots the DRT for each concentration in another plot
    o2_dependence, bool: (default = True)
        If True, plots the ln(1/ASR) vs. ln(O2 concentration) for the ohmic and polarization resistance
    drt_peaks, bool: (default = True)
        If True, the DRT peaks are fit and plotted. If the peaks have not been fit, they
        will be fit and added the the cell data excel sheet. If the data exists in the spreadsheet already
        then that data will be plotted.
    print_resistance_values, bool: (default = False)
        If True, prints the ohmic and polarization resistance for each concentration
    ncol, int: (default = 1)
        The number of columns to use for the plot legend 
    legend_loc, str: 
        The location of the legend (default = 'best'). The other option is to put 
        the legend outside the figure and this is done by setting legend_loc = 'outside'
    flow100, bool: (default = False)
        Set this to true if the total flow rate for the PO2 test was 100 SCCM
        for my older tests (cell 16 and below) my total flow rate was 100 SCCM
        This just changes the list of strings to look for
    cut_inductance, bool: (default = False)
        If this is set to true, the negative inductance values at the beginning of the DataFrame
    overwrite, bool: (default = False)
        If the data already exists in the cell excel file, then this will overwrite that data with
        the data currently being fit when overwrite=True.
        
    Return --> None, but it plots and shows one or more plots
    '''
    
    # +++++++++----- Finding and Dual DRT fitting the PO2 EIS
    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc
    substring_list = ['PEIS_20O2.80Ar','PEIS_40O2.60Ar','PEIS_60O2.40Ar','PEIS_80O2.20Ar','PEIS_100O2.0Ar'] # setting variables to look for
    O2_conc_list = [dta_files for dta_files in dta_files if any(sub in dta_files for sub in substring_list)] # placing all the changes in O2 eis files into a list
    cell_name = os.path.basename(fit_path).split("_", 2)[1]
    
    for peis in O2_conc_list:
        pickle_jar = os.listdir(fit_path)

        loc = os.path.join(folder_loc, peis) # creates the full path to the file
        po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name
        fit_name = cell_name + '_' + po2 + 'pO2'

        # +++++ checks to see if this has already been fit, if so then the eis will no be re-fit
        pickle_name = 0
        for pickle in pickle_jar: 
            file_name = fit_name + '_pfrt.pkl'
            if file_name == pickle:
                pickle_name = pickle_name + 1
                break
            
        if pickle_name == 0:
            pfrt_drt_save(loc,fit_path,fit_name,indicator=True)

    if flow100 == True: # Used for older cell data taken at 100 SCCM
        substring_list = ['PEIS_10O2.40Ar','PEIS_20O2.30Ar','PEIS_30O2.20Ar','PEIS_40O2.10Ar','PEIS_50O2.0Ar']

    O2_conc_list = sorted(O2_conc_list, key=lambda x: int((x[x.find('PEIS_')+len('PEIS_'):x.rfind('O2')]))) #Sorts numerically by PO2
    
    ' ---------- Plotting all the PO2 concentration EIS spectra'
    if eis == True:
        plt.figure() # Initializing the Figure

        for peis in O2_conc_list:
            # --- Finding and formatting
            loc = os.path.join(folder_loc, peis) # creates the full path to the file
            po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name

            nyquist_name =  po2 + '% O$_2$'

            if flow100 == True:
                po2_int = int(po2)
                po2 = str(po2_int * 2)
                nyquist_name = po2 + '% O$_2$'

            # --- Plotting
            plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc=legend_loc,cut_inductance = cut_inductance)

        plt.show()

    ' ---------- Plotting all the PO2 concentration EIS spectra'
    if drt == True: 
        fig, ax = plt.subplots() #initializing plots for DRT

        # --- Initializing lists for LnPO2
        O2_conc = np.array([]) # Concentration of Oxygen
        ohmic_asr = np.array([]) # ohm*cm^2
        rp_asr = np.array([]) # ohm*cm^2

        for peis in O2_conc_list: #For loop to plot the DRT of all PO2 files (probably a more elegant way, but this works)
            # --- Finding and formatting
            loc = os.path.join(folder_loc, peis) # creates the full path to the file
            po2 = peis[peis.find('PEIS_')+len('PEIS_'):peis.rfind('O2')] #gets the pO2 from the file name

            nyquist_name =  po2 + '% O$_2$'

            if flow100 == True:
                po2_int = int(po2)
                po2 = str(po2_int * 2)
                nyquist_name = po2 + '% O$_2$'

            # --- Plotting
            drt = DRT()

            fit_name = cell_name + '_' + po2 + 'pO2_pfrt.pkl'
            drt.load_attributes(os.path.join(fit_path,fit_name))

            label = po2+ '% O$_2$'
            drt.plot_distribution(label=label, return_line=True, ax=ax,area=area)

            # --- Appending resistance values to lists for lnPO2 plotting
            O2_conc = np.append(O2_conc,float(po2)/100)
            ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area)
            rp_asr = np.append(rp_asr,drt.predict_r_p() * area)

            # --- Printing the resistance values if that is desired
            if print_resistance_values == True:
                print(po2 + '% Oxygen Ohmic', round(drt.predict_r_inf() * area,3), '\u03A9 cm$^2^')
                print(po2 + '% Oxygen Rp', round(drt.predict_r_p() * area,3), '\u03A9 cm$^2^')
        
        # - Formatting
        ax.legend(fontsize='x-large')
        ax.xaxis.label.set_size('xx-large') 
        ax.yaxis.label.set_size('xx-large')
        ax.tick_params(axis='both', labelsize='x-large')
        
        plt.tight_layout()
        plt.show()

        'Creating a DataFrame and adding an excel data sheet (or creating the file if need be)'
        excel_name = '_' + cell_name + '_Data.xlsx'
        excel_file = os.path.join(folder_loc,excel_name)
        sheet_name = 'pO2_pfrt'
        exists = False

        exists, writer = excel_datasheet_exists(excel_file,sheet_name)

        if exists == False:
            df_po2 = pd.DataFrame(list(zip(O2_conc*100,ohmic_asr,rp_asr)),
                columns =['O2 Concentration (%)','Ohmic ASR (ohm*cm$^2$)', 'Rp ASR (ohm*cm$^2$)'])
            df_po2.to_excel(writer, sheet_name=sheet_name, index=False) # Writes this DataFrame to a specific worksheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

        elif exists == True and overwrite == True:
            df_po2 = pd.DataFrame(list(zip(O2_conc*100,ohmic_asr,rp_asr)),
                columns =['O2 Concentration (%)','Ohmic ASR (ohm*cm$^2$)', 'Rp ASR (ohm*cm$^2$)'])
            
            book = load_workbook(excel_file)
            writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_po2.to_excel(writer, sheet_name=sheet_name, index=False) # Extract data to an excel sheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

            df_po2 = pd.read_excel(excel_file,sheet_name)


    ' ------ Plotting the Oxygen dependence'
    if o2_dependence == True:
        lnpo2(ohmic_asr,rp_asr,O2_conc)
    elif drt==False and o2_dependence==True or print_resistance_values==True:
        print('Set drt to True. The cell resistance values are found using Jakes DRT package and thus the EIS spectra need to be fit')

    ' --- DRT peak fitting and plotting --- '
    if drt_peaks == True:
        o2_map_fits = [file for file in os.listdir(fit_path) if file.find('pO2')!=-1 and file.find('pfrt')!=-1] # gets all fuel cell map fits
        o2_map_fits = natsort.humansorted(o2_map_fits,key=lambda y: (len(y),y)) #sorts by bias
        
        # --- Checking to see if the peaks have already been fit:
        peak_data = False
        excel_name = '_' + cell_name + '_Data.xlsx'
        excel_file = os.path.join(folder_loc,excel_name)
        peak_data_sheet = 'pO2_PFRT_DRT_peak_fits'

        exists_peaks, writer_peaks = excel_datasheet_exists(excel_file,peak_data_sheet)
        
        if exists_peaks == False: # Make the excel data list
            # --- Fitting peaks and appending to a DataFrame
            df_tau_r = pd.DataFrame(columns = ['O2 Concentration (%)','Tau','Resistance']) #Initializing DataFrame to save temperature

            for fit in o2_map_fits: #Loading DRT, fitting peaks, and saving to a DataFrame
                # creating DRT object and loading fits
                drt = DRT()
                drt.load_attributes(os.path.join(fit_path,fit))


                # --- obtain time constants from inverters
                tau = drt.find_peaks() # τ/s
                r = np.array(drt.quantify_peaks()) * area # Ω*cm2

                # --- Obtaining pO2
                o2_conc = int(fit[fit.find(cell_name+'_')+len(cell_name+'_'):fit.rfind('pO2')])

                # Appending tau and r for each peak into df_tau_r
                i = 0
                for τ in tau:
                    df_tau_r.loc[len(df_tau_r.index)] = [o2_conc, τ, r[i]]
                    i = i+1

            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.

        elif exists == True and overwrite == False: #load the data into a DataFrame
            df_tau_r = pd.read_excel(excel_file,peak_data_sheet)

        elif exists_peaks == True and overwrite == True:
            # --- Fitting peaks and appending to a DataFrame
            df_tau_r = pd.DataFrame(columns = ['O2 Concentration (%)','Tau','Resistance']) #Initializing DataFrame to save temperature

            for fit in o2_map_fits: #Loading DRT, fitting peaks, and saving to a DataFrame
                # creating DRT object and loading fits
                drt = DRT()
                drt.load_attributes(os.path.join(fit_path,fit))


                # --- obtain time constants from inverters
                tau = drt.find_peaks() # τ/s
                r = np.array(drt.quantify_peaks()) * area # Ω*cm2

                # --- Obtaining pO2
                o2_conc = int(fit[fit.find(cell_name+'_')+len(cell_name+'_'):fit.rfind('pO2')])

                # Appending tau and r for each peak into df_tau_r
                i = 0
                for τ in tau:
                    df_tau_r.loc[len(df_tau_r.index)] = [o2_conc, τ, r[i]]
                    i = i+1
            
            book = load_workbook(excel_file)
            writer_peaks = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.

            df_tau_r = pd.read_excel(excel_file,peak_data_sheet)
        
        # ----- plotting
        palette = sns.color_palette('crest_r', as_cmap=True)
        plot = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df_tau_r, hue='O2 Concentration (%)',palette = palette,s=69)

        # ----- Aesthetics stuff
        sns.set_context("talk")
        fontsize = 14
        sns.despine()
        plot.set_ylabel('ASR (\u03A9 cm$^2$)',fontsize=fontsize)
        plot.set_xlabel('Time Constant (\u03C4/s)',fontsize=fontsize)
        plot.set(xscale='log')
        plt.tight_layout()
        plt.show()



def fc_bias_plots(folder_loc:str, fit_path:str, area:float, eis:bool=True, drt:bool=True,
                    drt_peaks:bool=True, ncol:int=1, legend_loc:str='best'):
    '''
    Finds all EIS files in the folder_loc taken during bias testing in Fuel Cell mode and plots the EIS
    The corresponding DRT fits are located and also plotted if drt = True
    If drt_peaks = True, the DRT spectra are fit, and the resistance of each DRT peak at each condition is plotted
    All data is append to a sheet in the cell data excel file if it does not already exist. If the data
    does already exist that data is called upon for plotting

    The .DTA EIS files are taken during a Gamry sequence that I use for testing the cell performance under
    various biases

    Parameters:
    -----------
    folder_loc, str:
        The folder location of the EIS files (path to directory)
    fit_path, str:
        The folder location of the DRT fits (path to directory)
    area, float: 
        The active cell area in cm^2
    eis, bool: (default = True)
        If True, the EIS spectra are plotted
    drt, bool: (default = True)
        If True, the DRT spectra are plotted
    drt_peaks, bool: (default = True)
        If True, the DRT peaks are fit and plotted. If the peaks have not been fit, they
        will be fit and added the the cell data excel sheet. If the data exists in the spreadsheet already
        then that data will be plotted.
    ncol, int: (default = 1)
        The number of columns in the legend
    legend_loc, str: (default = 'best')
        The location of the legend. The other option is to put 
        the legend outside the figure and this is done by setting legend_loc = 'outside'

    Return --> None but one or more plots are created and shown
    '''
    
    'Finding correct files and formatting'
    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc
    bias_eis = [] #initializing bias files list
    for file in dta_files: #Finding all fuel cell bias EIS files
        if (file.find('PEIS')!=-1) and (file.find('bias.DTA')!=-1) and (file.find('_n')!=-1):
            bias_eis.append(os.path.join(folder_loc,file))
    cell_name = os.path.basename(fit_path).split("_", 2)[1]
    bias_eis = sorted(bias_eis, key=lambda x: int((x[x.find('550C_n')+len('550C_n'):x.rfind('bias')]))) #Sorts numerically by bias

    for file in dta_files: #Finding the 0 bias condition and adding it to the beginning of the list of bias_eis
        if (file.find('PEIS')!=-1) and (file.find('0bias.DTA')!=-1): #adding in the 0 bias condition
            bias_eis.insert(0,os.path.join(folder_loc,file))

    'Plotting EIS'
    if eis == True:
        # --- Setting up the color map
        cmap = plt.cm.get_cmap('plasma') #cmr.redshift 
        color_space = np.linspace(0.2,0.8,len(bias_eis)) # array from 0-1 for the colormap for plotting
        c = 0 # index of the color array

        for peis in bias_eis:
            # --- Finding and formatting
            loc = os.path.join(folder_loc, peis) # creates the full path to the file
            bias = peis[peis.find('550C_')+len('PEIS_'):peis.rfind('bias')] #gets the bias from the file name
            if len(bias) > 1:
                bias = -int(bias[1:])/10

            nyquist_name =  str(bias) + 'V'

            # --- Plotting
            color = cmap(color_space[c])
            plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc=legend_loc,color=color)
            c = c+1
        plt.show()

    # 'Plotting DRT'
    # if drt == True:
    #     fig, ax = plt.subplots() #initializing plots for DRT

    #     # --- Setting up the color map
    #     cmap = plt.cm.get_cmap('plasma') #cmr.redshift 
    #     color_space = np.linspace(0.2,0.8,len(bias_eis)) # array from 0-1 for the colormap for plotting
    #     c = 0 # index of the color array

    #     # --- Initializing lists of data to save
    #     bias_array = np.array([]) # Applied bias of the eis spectra relative to OCV
    #     ohmic_asr = np.array([]) # ohm*cm^2
    #     rp_asr = np.array([]) # ohm*cm^2

    #     for peis in bias_eis: # For loop to plot the DRT of all PO2 files (probably a more elegant way, but this works)
    #         # --- Finding and formatting
    #         loc = os.path.join(folder_loc, peis) # creates the full path to the file
    #         bias = peis[peis.find('550C_')+len('PEIS_'):peis.rfind('bias')] #gets the bias from the file name
    #         map_fit_name = cell_name + '_map_fit_' + bias + 'bias.pkl'
    #         if len(bias) > 1:
    #             bias = -int(bias[1:])/10
                
    #         label =  str(bias) + 'V'

    #         # --- Updating arrays and plotting
    #         inv = Inverter()
    #         inv.load_fit_data(os.path.join(fit_path,map_fit_name))
    #         bias_array = np.append(bias_array,float(bias))
    #         ohmic_asr = np.append(ohmic_asr,inv.R_inf*area)
    #         rp_asr = np.append(rp_asr,inv.predict_Rp()*area)
    #         color = cmap(color_space[c])
    #         bp.plot_distribution(None,inv,ax,unit_scale='',label = label, color = color)
    #         c = c + 1
    #     ax.legend()
    #     plt.show()
        
    #     'Creating a DataFrame and adding an excel data sheet (or creating the file if need be)'
    #     excel_name = '_' + cell_name + '_Data.xlsx'
    #     excel_file = os.path.join(folder_loc,excel_name)
    #     sheet_name = 'FC mode data'
    #     exists = False

    #     if os.path.exists(excel_file)==True: # Looking for the data excel file in the button cell folder
    #         writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
    #         wb = load_workbook(excel_file, read_only=True) # Looking for the po2
    #         if sheet_name in wb.sheetnames:
    #             exists = True
    #     elif os.path.exists(excel_file)==False:
    #         writer = pd.ExcelWriter(excel_file,engine='xlsxwriter') #Creates a Pandas Excel writer using XlsxWriter as the engine. This will make a new excel file

    #     if exists == False:
    #         df_fc_bias = pd.DataFrame(list(zip(bias_array,ohmic_asr,rp_asr)),
    #             columns =['Applied Bias (V)','Ohmic ASR (ohm*cm^2)', 'Rp ASR (ohm*cm^2)'])
    #         df_fc_bias.to_excel(writer, sheet_name=sheet_name, index=False) # Writes this DataFrame to a specific worksheet
    #         writer.close() # Close the Pandas Excel writer and output the Excel file.

    # ' --- DRT peak fitting and plotting --- '
    # if drt_peaks == True:
    #     bias_map_fits = [file for file in os.listdir(fit_path) if (file.find('bias')!=-1) and (file.find('fit_n')!=-1 or file.find('0bias')!=-1)] # gets all fuel cell map fits
    #     bias_map_fits = natsort.humansorted(bias_map_fits,key=lambda y: (len(y),y)) #sorts by bias
    #     bias_map_fits.reverse()

    #     # --- Checking to see if the peaks have already been fit:
    #     peak_data = False
    #     excel_name = '_' + cell_name + '_Data.xlsx'
    #     excel_file = os.path.join(folder_loc,excel_name)
    #     peak_data_sheet = 'FC Bias DRT peak fits'

    #     if os.path.exists(excel_file)==True: # Looking for the data excel file in the button cell folder
    #         writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
    #         wb = load_workbook(excel_file, read_only=True) # Looking for the Arrhenius Data Sheet
    #         if peak_data_sheet in wb.sheetnames:
    #             peak_data = True
    #     elif os.path.exists(excel_file)==False:
    #         writer = pd.ExcelWriter(excel_file,engine='xlsxwriter') #Creates a Pandas Excel writer using XlsxWriter as the engine. This will make a new excel file
        
    #     if peak_data == False: # Make the excel data list
    #         # --- Fitting peaks and appending to a DataFrame
    #         df_tau_r = pd.DataFrame(columns = ['Bias','Tau','Resistance']) #Initializing DataFrame to save temperature
    #         for fit in bias_map_fits: #Loading DRT, fitting peaks, and saving to a DataFrame
    #             # creating inverter and calling fits
    #             inv = Inverter()
    #             inv.load_fit_data(os.path.join(fit_path,fit))
    #             inv.fit_peaks(prom_rthresh=0.05) # fit the peaks

    #             # --- obtain time constants from inverters
    #             tau = inv.extract_peak_info().get('tau_0') # τ/s
    #             r = inv.extract_peak_info().get('R')*area # Ω*cm2

    #             # --- Obtaining bias
    #             number = fit[fit.find('fit_n')+len('fit_n'):fit.rfind('bias')]
    #             try:
    #                 bias = -int(number)/10
    #             except ValueError:
    #                 bias = 0

    #             # Appending tau and r for each peak into df_tau_r
    #             i = 0
    #             for τ in tau:
    #                 df_tau_r.loc[len(df_tau_r.index)] = [bias, τ, r[i]]
    #                 i = i+1

    #         df_tau_r.to_excel(writer, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
    #         writer.close() # Close the Pandas Excel writer and output the Excel file.

    #     elif peak_data == True: #load the data into a DataFrame
    #         df_tau_r = pd.read_excel(excel_file,peak_data_sheet)

    #     # ----- plotting
    #     cmap = 'plasma_r'
    #     palette = sns.color_palette(cmap)
    #     plot = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df_tau_r, hue='Bias',palette = palette,s=69)

    #     # ----- Aesthetic stuff
    #     sns.set_context("talk")
    #     fontsize = 14
    #     sns.despine()
    #     plot.set_ylabel('ASR (\u03A9 cm$^2$)',fontsize=fontsize)
    #     plot.set_xlabel('Time Constant (\u03C4/s)',fontsize=fontsize)
    #     plot.set(xscale='log')
    #     plt.tight_layout()
    #     plt.show()



def o2_dual_drt_peaks(folder_loc:str, tau_low:float, tau_high:float, concs:np.array = None,
                        rmv_concs_r:np.array = None, rmv_concs_l:np.array = None,plot_all=False):
    '''
    This function is meant to linearly fit a single DRT peak across an oxygen concentration range
    This function can only be used after the po2_plots function
    This function plots ln(1/asr) vs. ln(O2 concentration (%))

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the directory containing the EIS files.
    tau_low, float:
        The lower bound of the time constant range to fit.
    tau_high, float:
        The upper bound of the time constant range to fit.
    concs, np.array: (default = None)
        The concentrations that the DRT peaks are taken at. if this is none all temperature ranges in
        the cell data sheet will be fit.
    rmv_concs_r, np.array: (default = None)
        If two clusters overlap, specify the concentrations where there are overlap and this will remove
        the peaks with higher time constants (lower frequency, to the right) from the fit.
    rmv_concs_l, np.array: (default = None)
        If two clusters overlap, specify the concentrations where there are overlap and this will remove
        the peaks with lower time constants (higher frequency, to the left) from the fit.

    Returns --> The slope of ln(1/asr)/ln(O2%), and a plot of the DRT peak fit and the activation energy is calculated and printed on the plot
    '''

    # ----- Sorting out the points from a specific time constant
    for file in os.listdir(folder_loc):
        if file.endswith('Data.xlsx'):
            data_file = os.path.join(folder_loc,file)
            break
    
    all_data = pd.read_excel(data_file,'pO2_dual_DRT_peaks')
    data = all_data[(all_data['Tau']>tau_low) & (all_data['Tau']<tau_high)]

    # ----- If desired, plotting all of the PO2 fits
    if plot_all == True:
        # ----- plotting
        palette = sns.color_palette('crest_r', as_cmap=True)
        plot = sns.scatterplot(x = 'Tau', y = 'Resistance', data = all_data, hue='O2 Concentration (%)',palette = palette,s=69)

        # ----- Ascetics stuff
        sns.set_context("talk")
        fontsize = 14
        sns.despine()
        plot.set_ylabel('ASR (\u03A9 cm$^2$)',fontsize=fontsize)
        plot.set_xlabel('Time Constant (\u03C4/s)',fontsize=fontsize)
        plot.set(xscale='log')

        plt.tight_layout()
        plt.show()

    # ----- If desired, only plotting certain temperatures
    if concs is not None:
        data = data[data['O2 Concentration (%)'].isin(concs)]

    # ----- removing a duplicate if there is overlap between clusters to the right (remove points to the right)
    if rmv_concs_r is not None:
        for conc in rmv_concs_r:
            # - find the lowest tau for a given temperature
            conc_data = data[data['O2 Concentration (%)']==conc]
            conc_data = conc_data.sort_values(by='Tau')

            try:
                highest_tau = conc_data.iloc[-1]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed concentration must be in concs array (aka concentrations plotted)')
                print('Check the concs array and the rmv_concs array')
                print('If that does not work, make sure that there are data points in the specified tau range (it is easy to confuse the log scale)')
                sys.exit(1)

            # - remove all higher tau values
            data = data[data['Tau']!=highest_tau]

    # ----- removing a duplicate if there is overlap between clusters to the left (remove points to the left)
    if rmv_concs_l is not None:
        for conc in rmv_concs_l:
            # - find the lowest tau for a given temperature
            conc_data = data[data['O2 Concentration (%)']==conc]
            conc_data = conc_data.sort_values(by='Tau')

            try:
                lowest_tau = conc_data.iloc[0]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed concentration must be in concs array (aka concentrations plotted)')
                print('Check the concs array and the rmv_concs array')
                print('If that does not work, make sure that there are data points in the specified tau range (it is easy to confuse the log scale)')
                sys.exit(1)

            # - remove all higher tau values
            data = data[data['Tau']!=lowest_tau]
    
    # ----- Finding low and high Tau values of the values plotted
    min_tau = data['Tau'].min()
    max_tau = data['Tau'].max()

    # ----- Formatting data for plotting
    rp_asr = data['Resistance'].values
    concs = data['O2 Concentration (%)'].values
    ln_rp_asr = np.log(1/rp_asr)
    
    o2_concs = np.array(concs)
    ln_o2 = np.log(o2_concs/100)

    # ----- Plotting
    x = ln_o2
    y = ln_rp_asr

    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax1.plot(x,y,'ko')

    # - Setting font sizes
    label_fontsize = 'x-large'
    tick_fontsize = 'large'
    text_fontsize = 'x-large'

    # - Setting ticks
    ax1.set_xticks(x, np.round(x,2), fontsize=tick_fontsize)
    ax1.set_yticks(y,labels= np.round(y,2), fontsize=tick_fontsize)
    ax2.set_xticks(x, labels = concs, fontsize=tick_fontsize)
    ax1.set_xlim(-1.7,0.1)
    ax2.set_xlim(-1.7,0.1)

    # - linear Fit:
    mr,br = np.polyfit(ln_o2,ln_rp_asr,1)
    mr,br, r, p_value, std_err = scipy.stats.linregress(x, y)
    fit_r = mr*ln_o2 + br
    ax1.plot(x, fit_r,'g')

    # - Axis labels:
    ax1.set_xlabel('ln(O$_2$) (%)',fontsize=label_fontsize)
    ax2.set_xlabel('Oxygen Concentration (%)',fontsize=label_fontsize)
    ax1.set_ylabel('ln(1/ASR) (S/cm$^2$)',fontsize=label_fontsize)

    # - Creating table:
    row_labels = ['r squared']
    slopes = f'{round(mr,2)}'
    table_values = [[round(r**2,3)]] # $\\bf{round(mr,2)}$
    color = 'palegreen'

    if mr >= 0:
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',
            rowColours= [color,color])
    else:
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower center',
            rowColours= [color,color])

    table.scale(1,1.6)
    
    # - Printing figure text
    mr_str = f'{round(mr,2)}'
    tau_lows = f'{min_tau:.2e}'
    tau_highs = f'{max_tau:.2e}'
    if mr >=0:
        fig.text(0.68,0.22,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold',fontsize = tick_fontsize)
        fig.text(0.17,0.81,'DRT peak between '+tau_lows+'(\u03C4/s) and '+tau_highs+'(\u03C4/s)',fontsize = tick_fontsize)
    else:
        fig.text(0.37,0.22,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold',fontsize = tick_fontsize)
        fig.text(0.38,0.81,'DRT peak between '+tau_lows+'(\u03C4/s) and '+tau_highs+'(\u03C4/s)',fontsize = tick_fontsize)
    plt.tight_layout()

    plt.show()

    return(mr)

def o2_pfrt_drt_peaks(folder_loc:str, tau_low:float, tau_high:float, concs:np.array = None,
                        rmv_concs_r:np.array = None, rmv_concs_l:np.array = None):
    '''
    This function is meant to linearly fit a single DRT peak across an oxygen concentration range
    This function can only be used after the po2_plots function
    This function plots ln(1/asr) vs. ln(O2 concentration (%))

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the directory containing the EIS files.
    tau_low, float:
        The lower bound of the time constant range to fit.
    tau_high, float:
        The upper bound of the time constant range to fit.
    concs, np.array: (default = None)
        The concentrations that the DRT peaks are taken at. if this is none all temperature ranges in
        the cell data sheet will be fit.
    rmv_concs_r, np.array: (default = None)
        If two clusters overlap, specify the concentrations where there are overlap and this will remove
        the peaks with higher time constants (lower frequency, to the right) from the fit.
    rmv_concs_l, np.array: (default = None)
        If two clusters overlap, specify the concentrations where there are overlap and this will remove
        the peaks with lower time constants (higher frequency, to the left) from the fit.

    Returns --> The slope of ln(1/asr)/ln(O2%), and a plot of the DRT peak fit and the activation energy is calculated and printed on the plot
    '''

    # ----- Sorting out the points from a specific time constant
    for file in os.listdir(folder_loc):
        if file.endswith('Data.xlsx'):
            data_file = os.path.join(folder_loc,file)
            break
    
    data = pd.read_excel(data_file,'pO2_PFRT_DRT_peak_fits')
    data = data[(data['Tau']>tau_low) & (data['Tau']<tau_high)]

    # ----- If desired, only plotting certain temperatures
    if concs is not None:
        data = data[data['O2 Concentration (%)'].isin(concs)]

    # ----- removing a duplicate if there is overlap between clusters to the right (remove points to the right)
    if rmv_concs_r is not None:
        for conc in rmv_concs_r:
            # - find the lowest tau for a given temperature
            conc_data = data[data['O2 Concentration (%)']==conc]
            conc_data = conc_data.sort_values(by='Tau')

            try:
                highest_tau = conc_data.iloc[-1]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed concentration must be in concs array (aka concentrations plotted)')
                print('Check the concs array and the rmv_concs array')
                print('If that does not work, make sure that there are data points in the specified tau range (it is easy to confuse the log scale)')
                sys.exit(1)

            # - remove all higher tau values
            data = data[data['Tau']!=highest_tau]

    # ----- removing a duplicate if there is overlap between clusters to the left (remove points to the left)
    if rmv_concs_l is not None:
        for conc in rmv_concs_l:
            # - find the lowest tau for a given temperature
            conc_data = data[data['O2 Concentration (%)']==conc]
            conc_data = conc_data.sort_values(by='Tau')

            try:
                lowest_tau = conc_data.iloc[0]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed concentration must be in concs array (aka concentrations plotted)')
                print('Check the concs array and the rmv_concs array')
                print('If that does not work, make sure that there are data points in the specified tau range (it is easy to confuse the log scale)')
                sys.exit(1)

            # - remove all higher tau values
            data = data[data['Tau']!=lowest_tau]
    
    # ----- Finding low and high Tau values of the values plotted
    min_tau = data['Tau'].min()
    max_tau = data['Tau'].max()

    # ----- Formatting data for plotting
    rp_asr = data['Resistance'].values
    concs = data['O2 Concentration (%)'].values
    ln_rp_asr = np.log(1/rp_asr)
    
    o2_concs = np.array(concs)
    ln_o2 = np.log(o2_concs/100)

    # ----- Plotting
    x = ln_o2
    y = ln_rp_asr

    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax1.plot(x,y,'ko')

    # - Setting font sizes
    label_fontsize = 'x-large'
    tick_fontsize = 'large'
    text_fontsize = 'x-large'

    # - Setting ticks
    ax1.set_xticks(x, np.round(x,2), fontsize=tick_fontsize)
    ax1.set_yticks(y,labels= np.round(y,2), fontsize=tick_fontsize)
    ax2.set_xticks(x, labels = concs, fontsize=tick_fontsize)
    ax1.set_xlim(-1.7,0.1)
    ax2.set_xlim(-1.7,0.1)

    # - linear Fit:
    mr,br = np.polyfit(ln_o2,ln_rp_asr,1)
    mr,br, r, p_value, std_err = scipy.stats.linregress(x, y)
    fit_r = mr*ln_o2 + br
    ax1.plot(x, fit_r,'g')

    # - Axis labels:
    ax1.set_xlabel('ln(O$_2$) (%)',fontsize=label_fontsize)
    ax2.set_xlabel('Oxygen Concentration (%)',fontsize=label_fontsize)
    ax1.set_ylabel('ln(1/ASR) (S/cm$^2$)',fontsize=label_fontsize)

    # - Creating table:
    row_labels = ['r squared']
    slopes = f'{round(mr,2)}'
    table_values = [[round(r**2,3)]] # $\\bf{round(mr,2)}$
    color = 'palegreen'

    if mr >= 0:
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',
            rowColours= [color,color])
    else:
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower center',
            rowColours= [color,color])

    table.scale(1,1.6)
    
    # - Printing figure text
    mr_str = f'{round(mr,2)}'
    tau_lows = f'{min_tau:.2e}'
    tau_highs = f'{max_tau:.2e}'
    if mr >=0:
        fig.text(0.68,0.22,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold',fontsize = tick_fontsize)
        fig.text(0.17,0.81,'DRT peak between '+tau_lows+'(\u03C4/s) and '+tau_highs+'(\u03C4/s)',fontsize = tick_fontsize)
    else:
        fig.text(0.37,0.22,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold',fontsize = tick_fontsize)
        fig.text(0.38,0.81,'DRT peak between '+tau_lows+'(\u03C4/s) and '+tau_highs+'(\u03C4/s)',fontsize = tick_fontsize)
    plt.tight_layout()

    plt.show()

    return(mr)

def excel_datasheet_exists(excel_file:str,sheet_name:str):
    '''
    Finds if the excel file exists and if the sheet name exists in the excel file
    If the excel file does not exist, it will return a writer that can create excel files
    If the excel file does exist, it will return a writer that can append to the excel file
    Will return whether or not the sheet_name exists in the excel file

    param excel_file,str: The excel file to look for
    param sheet_name,str: The name of the excel sheet to look for

    Return --> appropriate writer, whether or not the excel sheet exists
    '''
    
    exists = False
    if os.path.exists(excel_file)==True: # Looking for the data excel file in the button cell folder
        writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
        wb = load_workbook(excel_file, read_only=True) # Looking for the bias Deg eis
        
        if sheet_name in wb.sheetnames:
            exists = True

    elif os.path.exists(excel_file)==False:
        writer = pd.ExcelWriter(excel_file,engine='xlsxwriter')
    
    return exists,writer


















