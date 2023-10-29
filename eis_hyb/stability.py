''' This module contains functions to format, plot, analyze, and save EIS and DRT spectra for stability testing.
The data comes from a Gamry Potentiostat
# C-Meisel
'''

'Imports'
import os
import pandas as pd
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
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.ticker as ticker




from hybdrt.models import DRT, elements, drtbase
import hybdrt.plotting as hplt
from hybdrt.fileload import read_eis, get_eis_tuple, get_timestamp

from .plotting import plot_peis, plot_peiss, lnpo2
from .fit_drt import dual_drt_save, pfrt_drt_save
from .data_formatting import peis_data
from .convenience import excel_datasheet_exists, quick_dualdrt_plot, append_drt_peaks, rp_ohmic_to_excel, df_tau_r_to_excel


' List of Functions to Re-package:'
# - fcstb OCV eis plots (May not need to change these 4)
# - fcstb bias eis plots
# - ecstb OCV eis plots
# - ecstb bias eis plots
# - May need to re-re-make functions to save the fits (then I will never re-package again lol)


def fc_stb_ocv_eis(folder_loc:str, area:float, start_file:str = 'default',
                        a10:bool=True, eis:bool=True, drt:bool=True, 
                        resistance_plot:bool= True, drt_peaks:bool = True, ncol:int=1,
                        legend_loc:str='outside', cbar = True,
                        peaks_to_fit = 'best_id'):
    '''
    Plots the EIS and the fit DRT of a cell taken at OCV during a long term stability test. This function
    complements a Gamry sequence that I use to test the cell stability over time in fuel cell mode. 
    DRT must already be fit before using this function.

    OCV eis spectra are taken one an hour for the first 10 hours then 1 every 10 hours for the rest of the test
    
    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The folder location of the EIS files 
    area, float: 
        The active cell area in cm^2
    start_file, str: (default = 'default')
        The name of the first file taken in the stability sequence
    a10, bool: (After 10 hours) (default = True)
        Whether or not to plot the first 10 hours of the test.
        if True, then only hours 10 onward are plotted
    eis, bool: (default = True)
        Whether or not to plot the EIS spectra
    drt, bool: (default = True)
        Whether or not to plot the DRT fits
    resistance_plot, bool: (default = True)
        Whether or not to plot Ohmic, Rp, and Rtot over time
    drt_peaks, bool: (default = True)
        Whether or not to plot the DRT peak resistances overtime.
    ncol, int: (default = 1)
        The number of columns to use in the legend of the plot
    legend_loc, str:  (default = 'outside')
        The location of the legend (default = 'best'). The other option is to put 
        the legend outside the figure and this is done by setting legend_loc = 'outside'
    cbar, Bool: (default = True)
        If this is set to true then the legend will just be a colorbar
        If false, then the normal legend will exist
        As of now only applies to EIS
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amount of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer

    Return --> None but one or more plots are created and shown
    '''
    # --- Finding relavent information
    t0 = find_start_time(folder_loc = folder_loc, start_file = start_file)
    cell_name = find_cell_name(folder_loc)
        
    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc

    # --- Getting the end time
    a10_ocv_eis = select_stb_eis(folder_loc, dta_files, a10=True, bias = False, fc_operation = True)
    a10_ocv_eis.sort(key=lambda x:x[1])
    last = a10_ocv_eis[-1][1]
    end_time = int((last-t0)/3600) #hrs

    if eis == True:
        if a10 == False: #Plot the EIS for the first 10 hours in 1 hr increments (the increment it is taken in)
            f10_ocv_eis = select_stb_eis(folder_loc, dta_files, a10 = False, bias = False, fc_operation = True) # Finding correct EIS files and formatting

            'Plotting EIS'
            for peis in f10_ocv_eis:
                # --- Finding and formatting
                loc = os.path.join(folder_loc, peis) # creates the full path to the file
                time = peis[peis.find('__#')+len('__#'):peis.rfind('.DTA')] #gets the bias from the file name
                
                nyquist_name =  str(time) + ' Hours'

                # --- Plotting
                plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc='Outside')

            plt.show()

        if a10 == True: # Plot the DRT after 10 hours in 10 hour increments (the increment it is taken in)
            'Finding and formatting'
            a10_ocv_eis = select_stb_eis(folder_loc, dta_files, a10 = True, bias = False, fc_operation = True)

            # --- Getting the end time
            a10_ocv_eis.sort(key=lambda x:x[1])
            last = a10_ocv_eis[-1][1]
            end_time = int((last-t0)/3600) #hrs

            'Setting up an array for the color map'
            cmap = plt.cm.get_cmap('viridis', end_time)

            'Plotting EIS'
            fig,ax = plt.subplots()
            for peis in a10_ocv_eis:
                loc = os.path.join(folder_loc, peis[0]) # creates the full path to the file

                # --- Finding time of the EIS from the start of the degradation test
                test_time = peis[1]
                time = round((test_time-t0)/3600) #hrs
                nyquist_name =  str(time) + ' Hours'

                # --- Plotting
                df_useful = peis_data(area,loc)
                if cbar == True:
                    ax.plot(df_useful['ohm'],df_useful['ohm.1'],'o',markersize=9,label = nyquist_name,color = cmap(time)) #plots data

                if cbar == False:
                    plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc=legend_loc,color = cmap(time))


                # --- Plot Formatting                  
                ax.set_xlabel('Zreal (\u03A9 cm$^2$)',fontsize='xx-large')
                ax.set_ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',fontsize='xx-large')
                ax.tick_params(axis='both', which='major', labelsize='x-large')
                ax.axhline(y=0,color='k', linestyle='-.') # plots line at 0 #D2492A is the color of Mines orange
                ax.axis('scaled') #Keeps X and Y axis scaled 1 to 1
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if cbar == True:
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([0,end_time])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
                cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
                cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
                cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

    if drt == True:
        'Finding and Formatting'
        fig, ax = plt.subplots() #initializing plots for DRT

        if a10 == False: #Plot the DRT for the first 10 hours in 1 hr increments (the increment it is taken in)
            f10_ocv_eis = select_stb_eis(folder_loc, dta_files, a10 = False, bias = False, fc_operation = True)

            for eis in f10_ocv_eis: # Calling fits and plotting
                # --- Finding and formatting
                loc = os.path.join(folder_loc, eis) # creates the full path to the file
                time = eis[eis.find('__#')+len('__#'):eis.rfind('.DTA')] #gets the bias from the file name
                if time == 1:
                    label = str(time) + ' Hour'
                else:
                    label = str(time) + ' Hours'

                # --- Plotting
                quick_dualdrt_plot(loc, area, label=label, ax=ax, peaks_to_fit = peaks_to_fit, scale_prefix = "")    

            ax.legend()
            plt.show()
        
        if a10 == True:
            # --- Selecting eis at OCV after 10 hours of operation  
            a10_ocv_eis = select_stb_eis(folder_loc, dta_files, a10=True, bias = False, fc_operation = True)
            
            # --- Setting up an array for the color map
            cmap = plt.cm.get_cmap('viridis')
            color_space = np.linspace(0,1,len(a10_ocv_eis)) # array from 0-1 for the colormap for plotting
            c = 0 # index of the color array

            # --- Initializing lists for saving data
            time_list = np.array([]) # Concentration of Oxygen
            ohmic_asr = np.array([]) # ohm*cm^2
            rp_asr = np.array([]) # ohm*cm^2
            df_tau_r = pd.DataFrame(columns = ['Time (hrs)','Tau','Resistance']) #Initializing DataFrame to save temperature

            # --- Plotting
            for eis in a10_ocv_eis:
                # --- Extracting the time
                time = round((int(eis[1])-t0)/3600) # to convert to hours and round to the nearest hour from the test start
                label = str(time) + ' Hours'

                # --- Plotting
                print(eis[0])
                color = cmap(color_space[c])
                drt = quick_dualdrt_plot(eis[0], area, label=label, ax=ax, peaks_to_fit = peaks_to_fit,
                                   mark_peaks=False, scale_prefix = "", legend = False, color=color)  
                c = c + 1

                # --- Extracting data from the DRT fits and appending it to lists
                time_list = np.append(time_list,time) # hrs
                ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area) # Ω * cm^2
                rp_asr = np.append(rp_asr,drt.predict_r_p() * area) # Ω * cm^2
                append_drt_peaks(df_tau_r, drt, area, time, peaks_to_fit = 'best_id')

            # --- Appending Data to excel:
            rp_ohmic_to_excel(cell_name, folder_loc, ohmic_asr, rp_asr, 'dual_fc_stb_ocv', time_list, 'Time (Hrs)', overwrite = True)
            df_tau_r_to_excel(cell_name, folder_loc, df_tau_r,'dual_fc_stb_ocv_peaks', overwrite = True)

            # ax.legend(fontsize='small')
            # --- Colorbar formatting
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([0,end_time])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
            cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
            cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
            cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

    if resistance_plot == True:
        try:
            excel_name = '_' + cell_name + '_Data.xlsx'
            excel_file = os.path.join(folder_loc,excel_name)
            df = pd.read_excel(excel_file,'dual_fc_stb_ocv')
            df['Rtot (ohm*cm$^2$)'] =  df['Ohmic ASR (ohm*cm$^2$)'] + df['Rp ASR (ohm*cm$^2$)']

            # --- Setting variables for convenience
            x = df['Time (Hrs)']
            ohmic = df['Ohmic ASR (ohm*cm$^2$)']
            rp = df['Rp ASR (ohm*cm$^2$)']
            rtot = df['Rtot (ohm*cm$^2$)']

            # --- Getting the colors for the resistance values:
            cmap = cm.get_cmap('viridis')
            positions = [0, 0.5, 0.85]
            colors = [cmap(pos) for pos in positions]

            # --- Plotting
            fig, ax = plt.subplots()
            ax.scatter(x, ohmic, label='Ohmic', color = colors[0])
            ax.scatter(x, rp, label='Polarization', color = colors[1])
            ax.scatter(x, rtot, label = 'Total', color = colors[2])

            # --- Formatting
            ax.set_ylabel('ASR (\u03A9 cm$^2$)', size = 'xx-large')
            ax.set_xlabel('Time (Hrs)', size = 'xx-large')
            ax.legend(fontsize = 'large', frameon=False, handletextpad=0)
            ax.set_ylim(0)

            # --- Excessive formatting
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize='x-large')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            plt.tight_layout()
            plt.show()
        
        except ValueError as e:
            print(f"ValueError: {e}")
            print('This error is triggered because there is no resistance data found')
            print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
            print('Run this function again with DRT=True and with a10 = True')


    if drt_peaks == True:
        try:
            excel_name = '_' + cell_name + '_Data.xlsx'
            excel_file = os.path.join(folder_loc,excel_name)
            df = pd.read_excel(excel_file,'dual_fc_stb_ocv_peaks')

            # ----- plotting
            fig, ax = plt.subplots()
            cmap = plt.cm.get_cmap('viridis')
            ax = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df, hue='Time (hrs)',palette = cmap,s=69, legend=False)

            # ---- Formatting
            ax.set_ylabel('Peak ASR (\u03A9 cm$^2$)',fontsize='xx-large')
            ax.set_xlabel('Time Constant (\u03C4/s)',fontsize='xx-large')
            ax.set(xscale='log')
            ax.set_ylim(0)

            # --- Excessive formatting
            sns.despine()
            ax.tick_params(axis='both', which='major', labelsize='x-large')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            # --- Colorbar formatting
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([0,end_time])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
            cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
            cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
            cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

        except ValueError as e:
            print(f"ValueError: {e}")
            print('This error is triggered because there is no DRT peak data found')
            print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
            print('Run this function again with DRT=True and with a10 = True')

def fc_stb_bias_eis(folder_loc:str, area:float, start_file:str = 'default',
                        a10:bool=True, eis:bool=True, drt:bool=True, 
                        resistance_plot:bool= True, drt_peaks:bool = True, ncol:int=1,
                        legend_loc:str='outside', cbar = True,
                        peaks_to_fit = 'best_id'):
    '''
    Plots the EIS and the fit DRT of a cell taken at bias during a long term stability test. This function
    complements a Gamry sequence that I use to test the cell stability over time in fuel cell mode. 
    DRT must already be fit before using this function.

    Bias eis spectra are taken one an hour for the first 10 hours then 1 every 10 hours for the rest of the test
    
    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The folder location of the EIS files 
    area, float: 
        The active cell area in cm^2
    start_file, str: (default = 'default')
        The name of the first file taken in the stability sequence
    a10, bool: (After 10 hours) (default = True)
        Whether or not to plot the first 10 hours of the test.
        if True, then only hours 10 onward are plotted
    eis, bool: (default = True)
        Whether or not to plot the EIS spectra
    drt, bool: (default = True)
        Whether or not to plot the DRT fits
    resistance_plot, bool: (default = True)
        Whether or not to plot Ohmic, Rp, and Rtot over time
    drt_peaks, bool: (default = True)
        Whether or not to plot the DRT peak resistances overtime.
    ncol, int: (default = 1)
        The number of columns to use in the legend of the plot
    legend_loc, str:  (default = 'outside')
        The location of the legend (default = 'best'). The other option is to put 
        the legend outside the figure and this is done by setting legend_loc = 'outside'
    cbar, Bool: (default = True)
        If this is set to true then the legend will just be a colorbar
        If false, then the normal legend will exist
        As of now only applies to EIS
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amount of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer

    Return --> None but one or more plots are created and shown
    '''
    # --- Finding relavent information
    t0 = find_start_time(folder_loc = folder_loc, start_file = start_file)
    cell_name = find_cell_name(folder_loc)
        
    dta_files = [file for file in os.listdir(folder_loc) if file.endswith('.DTA')] #Makes a list of all .DTA files in the folder loc

    # --- Getting the end time
    a10_bias_eis = select_stb_eis(folder_loc, dta_files, a10=True, bias = True, fc_operation = True)
    a10_bias_eis.sort(key=lambda x:x[1])
    last = a10_bias_eis[-1][1]
    end_time = int((last-t0)/3600) #hrs

    if eis == True:
        if a10 == False: #Plot the EIS for the first 10 hours in 1 hr increments (the increment it is taken in)
            f10_bias_eis = select_stb_eis(folder_loc, dta_files, a10 = False, bias = True, fc_operation = True) # Finding correct EIS files and formatting

            'Plotting EIS'
            for peis in f10_bias_eis:
                # --- Finding and formatting
                loc = os.path.join(folder_loc, peis) # creates the full path to the file
                time = peis[peis.find('__#')+len('__#'):peis.rfind('.DTA')] #gets the bias from the file name
                
                nyquist_name =  str(time) + ' Hours'

                # --- Plotting
                plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc='Outside')

            plt.show()

        if a10 == True: # Plot the DRT after 10 hours in 10 hour increments (the increment it is taken in)
            'Finding and formatting'
            a10_bias_eis = select_stb_eis(folder_loc, dta_files, a10 = True, bias = True, fc_operation = True)

            'Setting up an array for the color map'
            cmap = plt.cm.get_cmap('plasma', end_time)

            'Plotting EIS'
            fig,ax = plt.subplots()
            for peis in a10_bias_eis:
                loc = os.path.join(folder_loc, peis[0]) # creates the full path to the file

                # --- Finding time of the EIS from the start of the degradation test
                test_time = peis[1]
                time = round((test_time-t0)/3600) #hrs
                nyquist_name =  str(time) + ' Hours'

                # --- Plotting
                df_useful = peis_data(area,loc)
                if cbar == True:
                    ax.plot(df_useful['ohm'],df_useful['ohm.1'],'o',markersize=9,label = nyquist_name,color = cmap(time)) #plots data

                if cbar == False:
                    plot_peiss(area,nyquist_name,loc,ncol=ncol,legend_loc=legend_loc,color = cmap(time))


                # --- Plot Formatting                  
                ax.set_xlabel('Zreal (\u03A9 cm$^2$)',fontsize='xx-large')
                ax.set_ylabel('$\u2212$Zimag (\u03A9 cm$^2$)',fontsize='xx-large')
                ax.tick_params(axis='both', which='major', labelsize='x-large')
                ax.axhline(y=0,color='k', linestyle='-.') # plots line at 0 #D2492A is the color of Mines orange
                ax.axis('scaled') #Keeps X and Y axis scaled 1 to 1
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if cbar == True:
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([0,end_time])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
                cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
                cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
                cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

    if drt == True:
        'Finding and Formatting'
        fig, ax = plt.subplots() #initializing plots for DRT

        if a10 == False: #Plot the DRT for the first 10 hours in 1 hr increments (the increment it is taken in)
            f10_bias_eis = select_stb_eis(folder_loc, dta_files, a10 = False, bias = True, fc_operation = True)

            for eis in f10_bias_eis: # Calling fits and plotting
                # --- Finding and formatting
                loc = os.path.join(folder_loc, eis) # creates the full path to the file
                time = eis[eis.find('__#')+len('__#'):eis.rfind('.DTA')] #gets the bias from the file name
                if time == 1:
                    label = str(time) + ' Hour'
                else:
                    label = str(time) + ' Hours'

                # --- Plotting
                quick_dualdrt_plot(loc, area, label=label, ax=ax, peaks_to_fit = peaks_to_fit, scale_prefix = "")    

            ax.legend()
            plt.show()
        
        if a10 == True:
            # --- Selecting eis at OCV after 10 hours of operation  
            a10_bias_eis = select_stb_eis(folder_loc, dta_files, a10=True, bias = True, fc_operation = True)
            
            # --- Setting up an array for the color map
            cmap = plt.cm.get_cmap('plasma')
            color_space = np.linspace(0,1,len(a10_bias_eis)) # array from 0-1 for the colormap for plotting
            c = 0 # index of the color array

            # --- Initializing lists for saving data
            time_list = np.array([]) # Concentration of Oxygen
            ohmic_asr = np.array([]) # ohm*cm^2
            rp_asr = np.array([]) # ohm*cm^2
            df_tau_r = pd.DataFrame(columns = ['Time (hrs)','Tau','Resistance']) #Initializing DataFrame to save temperature

            # --- Plotting
            for eis in a10_bias_eis:
                # --- Extracting the time
                time = round((int(eis[1])-t0)/3600) # to convert to hours and round to the nearest hour from the test start
                label = str(time) + ' Hours'

                # --- Plotting
                print(eis[0])
                color = cmap(color_space[c])
                drt = quick_dualdrt_plot(eis[0], area, label=label, ax=ax, peaks_to_fit = peaks_to_fit,
                                   mark_peaks=False, scale_prefix = "", legend = False, color=color)  
                c = c + 1

                # --- Extracting data from the DRT fits and appending it to lists
                time_list = np.append(time_list,time) # hrs
                ohmic_asr = np.append(ohmic_asr,drt.predict_r_inf() * area) # Ω * cm^2
                rp_asr = np.append(rp_asr,drt.predict_r_p() * area) # Ω * cm^2
                append_drt_peaks(df_tau_r, drt, area, time, peaks_to_fit = 'best_id')

            # --- Appending Data to excel:
            rp_ohmic_to_excel(cell_name, folder_loc, ohmic_asr, rp_asr, 'dual_fc_stb_bias', time_list, 'Time (Hrs)', overwrite = True)
            df_tau_r_to_excel(cell_name, folder_loc, df_tau_r,'dual_fc_stb_bias_peaks', overwrite = True)

            # --- Colorbar formatting
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([0,end_time])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
            cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
            cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
            cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

    if resistance_plot == True:
        try:
            excel_name = '_' + cell_name + '_Data.xlsx'
            excel_file = os.path.join(folder_loc,excel_name)
            df = pd.read_excel(excel_file,'dual_fc_stb_bias')
            df['Rtot (ohm*cm$^2$)'] =  df['Ohmic ASR (ohm*cm$^2$)'] + df['Rp ASR (ohm*cm$^2$)']

            # --- Setting variables for convenience
            x = df['Time (Hrs)']
            ohmic = df['Ohmic ASR (ohm*cm$^2$)']
            rp = df['Rp ASR (ohm*cm$^2$)']
            rtot = df['Rtot (ohm*cm$^2$)']

            # --- Getting the colors for the resistance values:
            cmap = cm.get_cmap('plasma')
            positions = [0, 0.5, 0.85]
            colors = [cmap(pos) for pos in positions]

            # --- Plotting
            fig, ax = plt.subplots()
            ax.scatter(x, ohmic, label='Ohmic', color = colors[0])
            ax.scatter(x, rp, label='Polarization', color = colors[1])
            ax.scatter(x, rtot, label = 'Total', color = colors[2])

            # --- Formatting
            ax.set_ylabel('ASR (\u03A9 cm$^2$)', size = 'xx-large')
            ax.set_xlabel('Time (Hrs)', size = 'xx-large')
            ax.legend(fontsize = 'large', frameon=False, handletextpad=0)
            ax.set_ylim(0)

            # --- Excessive formatting
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize='x-large')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            plt.tight_layout()
            plt.show()
        
        except ValueError as e:
            print(f"ValueError: {e}")
            print('This error is triggered because there is no resistance data found')
            print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
            print('Run this function again with DRT=True and with a10 = True')


    if drt_peaks == True:
        try:
            excel_name = '_' + cell_name + '_Data.xlsx'
            excel_file = os.path.join(folder_loc,excel_name)
            df = pd.read_excel(excel_file,'dual_fc_stb_bias_peaks')

            # ----- plotting
            fig, ax = plt.subplots()
            cmap = plt.cm.get_cmap('plasma')
            ax = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df, hue='Time (hrs)',palette = cmap,s=69, legend=False)

            # ---- Formatting
            ax.set_ylabel('Peak ASR (\u03A9 cm$^2$)',fontsize='xx-large')
            ax.set_xlabel('Time Constant (\u03C4/s)',fontsize='xx-large')
            ax.set(xscale='log')
            ax.set_ylim(0)

            # --- Excessive formatting
            sns.despine()
            ax.tick_params(axis='both', which='major', labelsize='x-large')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            # --- Colorbar formatting
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([0,end_time])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(sm,ticks = [0,end_time],cax=cax)
            cb.set_label(label='Time (hrs)',fontsize = 'xx-large',labelpad = -10)
            cb.set_ticks(ticks = [0,end_time],fontsize='xx-large')
            cb.ax.tick_params(labelsize='x-large')

            plt.tight_layout()
            plt.show()

        except ValueError as e:
            print(f"ValueError: {e}")
            print('This error is triggered because there is no DRT peak data found')
            print('Most likely, the DRT has not yet been fit to the EIS and analyzed')
            print('Run this function again with DRT=True and with a10 = True')

# > > > > > Functions to aid in the main stability functions
def find_start_time(folder_loc:str, start_file:str = 'default'):
    '''
    Finds the start time, in seconds from epoch, of the stability test

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The folder location of the EIS files
    start_file, str: (default = 'default')
        The name of the first file taken in the stability sequence

    Return --> t0, int, the start time in seconds from epoch

    '''
    if start_file == 'default': #if another file is specified as the first file, this file will be used to find T0
        for file in os.listdir(folder_loc): #Finding the first file
            if file.find('Deg__#1.DTA')!=-1 and file.find('OCV')!=-1:
                start_file = os.path.join(folder_loc,file)
        t0 = int(get_timestamp(start_file).strftime("%s")) # Getting time stamp for first file in s from epoch, and convert to int
    else:
        t0 = int(get_timestamp(start_file).strftime("%s")) # Getting time stamp for first file in s from epoch, and convert to int

    return t0

def select_stb_eis(folder_loc:str, dta_files:list, a10:bool = True, bias = True, fc_operation = True):
    '''
    Selects the specific eis spectra to be plot or inverted to DRT.
    This function also sorts the spectra by the time they were taken

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The folder location of the EIS files 
    dta_files, list:
        list of all .DTA files in the cell data folder
    a10, bool: (After 10 hours) (default = True)
        Whether or not to get the eis data from the the first 10 hours of the test.
        if True, then only hours 10 and onward are selected
    bias, bool: (default = True)
        Whether or not to select the bias eis
        if bias = False, then OCV data is selected
    fc_operation, bool: (default = True)
        whether or not to attain fuel cell mode stability data
        if fc = False, then electrolysis cell mode stability eis data will be selected

    Return --> tuple list, (file name, time)
    '''
    bias_eis = []
    stb_eis = [] #initializing the eis files list

    # --- Sorting OCV and bias plots
    if bias == True and fc_operation == True:
        for file in dta_files:
            if (file.find('PEIS')!=-1) and (file.find('n3Bias')!=-1):
                bias_eis.append(file)

    if bias == True and fc_operation == False:
        for file in dta_files:
            if (file.find('PEIS')!=-1) and (file.find('TNV')!=-1):
                bias_eis.append(file)

    if bias == False:
        for file in dta_files:
            if (file.find('PEIS')!=-1) and (file.find('n3Bias')==-1):
                bias_eis.append(file)


    # --- Sorting a10 and fc/ec plots
    if a10 == True and fc_operation == True: # Selecting for all a10 fuel cell eis files
        for file in bias_eis:
            if (file.find('_Deg_')==-1) and (file.find('Deg10')!=-1):
                loc = os.path.join(folder_loc,file)
                stb_eis.append((loc,int(get_timestamp(loc).strftime("%s"))))
        stb_eis.sort(key=lambda x:x[1])
        sorted_stb_eis = stb_eis

        
    elif a10 == False and fc_operation == True: # Selecting for all f10 fuel cell eis files
        for file in bias_eis: 
            if (file.find('_Deg_')!=-1) and (file.find('Deg10')==-1):
                stb_eis.append(file)
        sorted_stb_eis = sorted(stb_eis, key=lambda x: int((x[x.find('__#')+len('__#'):x.rfind('.DTA')]))) #Sorts numerically by time

    
    elif a10 == True and fc_operation == False: # Selecting for all a10 electrolysis cell eis files
        for file in bias_eis: 
            if (file.find('_ECstability')==-1) and (file.find('ECstability10')!=-1):
                loc = os.path.join(folder_loc,file)
                stb_eis.append((loc,int(get_timestamp(loc).strftime("%s"))))
        stb_eis.sort(key=lambda x:x[1])
        sorted_stb_eis = stb_eis
            
    elif a10 == False and fc_operation == False: # Selecting for all f10 electrolysis cell eis files
        for file in bias_eis:
            if (file.find('_ECstability')!=-1) and (file.find('ECstability10')==-1):
                stb_eis.append(file)
        sorted_stb_eis = sorted(stb_eis, key=lambda x: int((x[x.find('__#')+len('__#'):x.rfind('.DTA')]))) #Sorts numerically by time

    
    return(sorted_stb_eis)

def find_cell_name(folder_loc):
    """
    Finds the name of the cell

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The folder location of the EIS files 

    Return --> str, The name of the cell
    """
    for file in os.listdir(folder_loc): # Finding the name of the cell
        if file.find('PEIS')!=-1:
            cell_name = os.path.basename(file).split("_", 1)[0]
            break

    return(cell_name)