''' 
This module contains functions to help format and plot Arrhenius data.
The data is electrohchemical impedance spectroscopy (EIS) data obtained by a Gamry potentiostat. 
The files are .DTA files and this module plots EIS and IV curves as well as fits and plots DRT using
the Hybrid_DRT package developed by Dr. Jake Huang
# C-Meisel
'''

'Imports'
import os #operating system useful for navigating directories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import scipy as scipy
import seaborn as sns
import sys
import traceback

from hybdrt.models import DRT, elements, drtbase
import hybdrt.plotting as hplt
from hybdrt.fileload import read_eis, get_eis_tuple

from .plotting import plot_peiss, plot_ivfcs
from .convenience import excel_datasheet_exists, append_drt_peaks
from .data_formatting import peis_data

' Plotting params '
spine_thickness = 1.3
txt_spine_color = '#212121'  #'#212121' # 'black' #333333
plt.rcParams.update({
    'axes.linewidth': spine_thickness,        # Spine thickness
    'xtick.major.width': spine_thickness,     # Major tick thickness for x-axis
    'ytick.major.width': spine_thickness,     # Major tick thickness for y-axis
    'xtick.major.size': spine_thickness * 3,      # Major tick length for x-axis
    'ytick.major.size': spine_thickness * 3,      # Major tick length for y-axis
    
    'axes.grid': False,           # Add grid lines
    'grid.alpha': 0.4,           # Grid lines transparency
    
    'font.family': 'sans-serif',

    'text.color': txt_spine_color,                   # Color for all text
    'axes.labelcolor': txt_spine_color,              # Color for axis labels
    'axes.edgecolor': txt_spine_color,               # Color for axis spines
    'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
    'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
})


def arrhenius_plots_dual(folder_loc:str, temps:list, area:float=0.5, plot_eis:bool = True, plot_drt:bool = True,
                    drt_peaks:bool = True, thickness:float = 0, rp_plt_type:str = 'ln', re_fit:bool = False,
                    legend_loc:str = 'outside', drtp_leg_loc_ots:bool = False, reverse = False, peaks_to_fit:int = 'best_id',
                    drt_model:str='dual',save_figures:str=None):
    '''
    Searches though the folder_loc and separates out the EIS files to be used to plot the arrhenius data.
    The EIS files are matched to their corresponding temperature input from temps.
    The EIS files are dual-fit using the Hybrid-DRT packange by Dr. Jake Huang and the ohmic and rp values are extracted and used
    to make an Arrhneius plot for the ohmic and polarization resisitance. Each curve is linearly fit and the 
    activation energy of is calculated and shown on the plot. The data is then saved in the cell data excel
    file if it does not already exist. If the data does exist then that data will be used (unless re-fit = True).

    This function gives the option to plot the EIS, DRT, and DRT peaks for the Arrhenius EIS files.

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the directory containing the EIS files. 
    temps, list: 
        List of temperatures that the EIS spectra were taken at. this is input by the user 
        and must be in the same order that the EIS spectra were taken at.
    area, float: (Default = 0.5)
        The active cell area of the cell in cm^2
    plot_eis, bool: (Default = True)
        If true all the EIS spectra used in making the Arrhenius plot will be plotted.
    plot_drt, bool: (Default = True)
        If true the DRT map-fits will be plotted.
    drt_peaks, bool: (Default = True)
        If true all the DRT spectra will be fit using the Bayes-DRT package and the
        resistance of each peak will be plotted.
    thickness, float: (Default is 0)
        The thickness of the electrolyte of the cell in cm. If there is no SEM data on the electrolyte
        thickness then leave the value at it's default value of 0.
    rp_plt_type, str: (Default is 'ln')
        Type of arrhenius plot to make for the polarization resistance plot. The two values are ln or asr.
        The ln plot is a log-log plot and the asr plot is a semilogx plot. The activation energy (Ea) is 
        only calculated if an ln plot is made.
    re-fit, bool: optional (default: False)
        If True, the EIS data will have the DRT fits re-fit and re-stored in the cell data excel file.
        To reiterate this will overwrite the data in the data_sheet
    legend_loc, str: optional (default: 'outside')
        The location of the legend. Outside placed the legend outside the figure.
        The other option is 'best' and this uses 'best' to make the best decision on where to place
        the legend insisde the figure.
    drtp_leg_loc_ots, bool: optional (default: False)
        drt peak legend location outside the figure
        If True, the legend for the DRT peaks will be placed outside the figure.
    reverse, bool: optional (default: False)
        If the arrhenius plot was taken in reverse order i.e 500-625C (like the older and newer), then set this to true
        This reverses the direction of the cmap to keep the lower temps blue and higher temps red on the DRT and Nyquist plots
        To reiterate if the arrhenisu plot was taken from lowest temp to highest set this to true
    peaks_to_fit, str/int: (default: 'best_id')
        Sets the number of discrete elements in the EIS to fit in the DRT
        This basically just sets the amout of peaks to fit
        if this is set to the default 'best_id' then it fits the number of peaks suggested by dual_drt
        if this is set to a integer, it fit the number of peaks set by that integer
    drt_model, str: (default = dual)
        which type of drt used to analyze the data
        if drt = 'dual' the dual regression model is used to fit the data
        if drt = 'drtdop' then the drtdop model is used to fit the data
    save_figures, str: (default = None)
        If this is not none, all 4 figures (EIS, DRT, Ahp_O, and Ahp_Rp) will be saved.
        Save_eis is the file name and path of the folder to save the figures.
    Return --> None but 2-5 plots are crated and shown, the EIS data gets fit and saved, and the 
    data to make the Arrhenius plots is saved in the cell data excel file (if not already there)
    '''

    ' --- Finding correct files and formatting --- '
    ahp_eis = [file for file in os.listdir(folder_loc) if file.endswith('.DTA') and file.find('_Ahp__#')!=-1 and file.find('PEIS')!=-1] #Makes a list of all ahp
    cell_name = os.path.basename(ahp_eis[0]).split("_", 1)[0]
    ahp_eis = sorted(ahp_eis, key=lambda x: int((x[x.find('Ahp__#')+len('Ahp__#'):x.rfind('.DTA')]))) #Sorts numerically by eis number (temperature)
    
    if reverse == True:
        ahp_eis.reverse()
        temps.reverse()

    if reverse == False:
        temps.reverse()

    ' --- Gathering information to make the Arrhenius Plots and saving it in an excel file sheet for this cell --- '
    # --- Checking to see if this has already been done before
    excel_name = '_' + cell_name + '_Data.xlsx'
    excel_file = os.path.join(folder_loc,excel_name)
    sheet_name = 'Arrhenius data'
    exists = False

    exists, writer = excel_datasheet_exists(excel_file,sheet_name)

    if plot_drt == True and (exists == True or re_fit == False):
        print('The DRT needs to be fit or re-fit in order to plot it, set re_fit to True')

    # --- Initializing lists
    ohmic = np.array([]) # Ω
    rp = np.array([]) # Ω
    tk_1000 = np.array([]) #(1/k) 1000 over temperature in K.  From 625-500C (same order as above lists)
    df_tau_r = pd.DataFrame(columns = ['Temperature (C)','Tau','Resistance']) #Initializing DataFrame to save temperature
    ah_cond = np.array([]) # S/cm
    ah_ohmic_asr = np.array([]) # ohm*cm^2
    rp_asr = np.array([]) # ohm*cm^2
    ah_rp = np.array([]) # ohm*cm^2/T

    if exists == False or re_fit == True: # Fit the data and make the excel data list
        if plot_drt == True:
            fig, ax = plt.subplots()
            # - Setting COlormap
            cmap = plt.cm.get_cmap('coolwarm_r') #cmr.redshift
            color_space = np.linspace(0,1,len(ahp_eis)) # array from 0-1 for the colormap for plotting
            c = 0 # indicie of the color array

        for c,eis in enumerate(ahp_eis): # Dual DRT Inverting all ahp EIS data
            # - Extracting the temperature data
            temp = str(temps[c])
            label = temp + ' \u00B0C'

            # ----- Creating DRT instance and prepping EIS for inversion
            full_loc = os.path.join(folder_loc,eis)
            drt = DRT()
            df = read_eis(full_loc)
            freq,z = get_eis_tuple(df) # Get relavent data from EIS dataframe

            if drt_model == 'dual':
                drt.dual_fit_eis(freq, z, discrete_kw=dict(prior=True, prior_strength=None)) # Fit the data
                
                tau = drt.get_tau_eval(20)

                # - Selecting the number of peaks for the drt distribution to have.
                if peaks_to_fit == 'best_id':
                    best_id = drt.get_best_candidate_id('discrete', criterion='lml-bic')
                    peaks = best_id

                else: peaks = peaks_to_fit

                # - Selecting the model. The number of peaks to plot and fit
                model_dict = drt.get_candidate(peaks,'discrete')
                model = model_dict['model']            

                # ----- Setting up DRT plot if desired
                if plot_drt == True:
                    label = label
                    color = cmap(color_space[c])
                    mark_peaks_kw = {'color':color,'find_peaks_kw':{'method':'prob'}}

                    model.plot_distribution(tau, ax=ax, area=area,label=label,mark_peaks=True,color=color,mark_peaks_kw=mark_peaks_kw)
                    c = c + 1

                # --- obtain time constants from inverters and Appending tau and r for each peak into df_tau_r
                tau = model_dict['peak_tau'] # τ/s
            
            elif drt_model == 'drtdop':
                # -- Initializing drt instance and dop model
                drt.fit_dop=True

                # -- Fitting DRT
                drt.fit_eis(freq, z)    
                tau = drt.get_tau_eval(20)

                # -- Plotting
                if plot_drt == True:
                    color = cmap(color_space[c])
                    mark_peaks_kw = {'color':color,'sizes':[75]}
                    kwargs = {
                        'linewidth': 2,
                    }
                    drt.plot_distribution(tau, ax=ax, area=area,label=label,mark_peaks=True,
                                    scale_prefix="", c=color, plot_ci=False,
                                      mark_peaks_kw=mark_peaks_kw,**kwargs)
                    
            else:
                print('In order to plot DRT, it must be fit with the dual or the drtdop model') 
                print('Set drt= \'dual\' or to \'drtdop\'') 


            # ----- Extracting and calculating resistance data
            ohmic = np.append(ohmic,drt.predict_r_inf())
            rp = np.append(rp,drt.predict_r_p())
            tk_1000 = np.append(tk_1000, 1000/(int(temp)+273))
            append_drt_peaks(df_tau_r, drt, area, temp, peaks_to_fit = peaks_to_fit,
                             drt_model = drt_model)


        if plot_drt == True:
            # - Adding Frequency scale:
            def Tau_to_Frequency(T):
                return 1 / (2 * np.pi * T)

            freq_ax = ax.secondary_xaxis('top', functions=(Tau_to_Frequency, Tau_to_Frequency))
            
            # - Excessive formatting
            ax_title_size = 23
            tick_label_size = ax_title_size * 0.8
            legend_txt_size = ax_title_size * 0.7

            ax.legend(fontsize=legend_txt_size,frameon=False,handletextpad=0.5,
                    handlelength=1)

            ax.xaxis.label.set_size(ax_title_size) 
            ax.yaxis.label.set_size(ax_title_size)
            ax.tick_params(axis='both', labelsize=tick_label_size)
            ax.spines['right'].set_visible(False)

            freq_ax.set_xlabel('$f$ (Hz)',size=ax_title_size)
            freq_ax.tick_params(axis='x',labelsize=tick_label_size)
            
            plt.tight_layout()

            if save_figures is not None:
                drt_fig_name = os.path.join(save_figures,'Ahp_DRT.png')
                fmat = drt_fig_name.split('.', 1)[-1]
                fig.savefig(drt_fig_name, dpi=300, format=fmat, bbox_inches='tight') 

            plt.show()


        # - Calculating Ohmic and Polarization area specific resistance
        rp_asr = np.array(rp) * area #ohms*cm^2
        ohmic_asr = np.array(ohmic) * area #ohms*cm^2

        ' >>>>>>>>>> Calculating all relavent information for the Arrhenius Plots and making a table'
        # - Calculations
        conductivity = thickness/(ohmic_asr) #This is the conductivity of the electroltye
        ah_cond = np.log(conductivity*(273+np.array(temps))) #The temperature for this step is kelvin, thus 273 is added to the temperatue in celsius
        ah_rp = np.log(rp_asr/(273+np.array(temps)))
        ah_ohmic_asr = np.log(ohmic_asr/(273+np.array(temps)))

        # - Making a table with all relavent information
        df_table = pd.DataFrame(list(zip(temps, ohmic, conductivity, ohmic_asr, ah_ohmic_asr,
            ah_cond, rp_asr, ah_rp , tk_1000)), 
            columns =['Temperature (C)','Ohmic Resistance (ohm)', 'Conductivity (S/cm)', 'Ohmic ASR (ohm*cm^2)', 'ln(ohmic*cm^2/T)',
            'ln(sigma*T) (SK/cm)', 'Polarization Resistance ASR (ohm*cm^2)', 'ln(ohm*cm^2/T)','tk_1000 (1000/k)']) #\u03C3

        # - Saving the table to an excel sheet
        if exists == False:
            df_table.to_excel(writer, sheet_name=sheet_name, index=False) # Writes this dataframe to a specific worksheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

        elif exists == True:
            writer = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_table.to_excel(writer, sheet_name=sheet_name, index=False) # Extract data to an excel sheet
            writer.close() # Close the Pandas Excel writer and output the Excel file.

            df_table = pd.read_excel(excel_file,sheet_name)

        ' >>>>>>>>>> Appending the Peak_fit data to an excel file '
        peak_data_sheet = 'Ahp_' + drt_model + '_DRT_peaks'
        exists_peaks = False

        exists_peaks, writer_peaks = excel_datasheet_exists(excel_file,peak_data_sheet)
    
        if exists_peaks == False: # Make the excel data list
            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.

        elif exists_peaks == True and re_fit == True:
            writer_peaks = pd.ExcelWriter(excel_file,engine='openpyxl',mode='a',if_sheet_exists='replace') #Creates a Pandas Excel writer using openpyxl as the engine in append mode
            df_tau_r.to_excel(writer_peaks, sheet_name=peak_data_sheet, index=False) # Extract data to an excel sheet
            writer_peaks.close() # Close the Pandas Excel writer and output the Excel file.

            df_tau_r = pd.read_excel(excel_file,peak_data_sheet)


    elif exists == True: # Importing the dataframe with the lists if it does already exist and initializing lists
        df = pd.read_excel(excel_file,sheet_name)
        
        # --- initializing lists
        tk_1000 = df['tk_1000 (1000/k)'].values
        ah_cond = df['ln(sigma*T) (SK/cm)'].values
        ah_ohmic_asr = df['ln(ohmic*cm^2/T)'].values
        rp_asr = df['Polarization Resistance ASR (ohm*cm^2)'].values
        ah_rp = df['ln(ohm*cm^2/T)'].values
        ohmic = df['Ohmic Resistance (ohm)'].values

    ' --- Making the Ohmic Resistance Arrhenius Plot --- '
    # - General formatting
    ax_title_fs = 18
    ax_tl_fs = ax_title_fs * 0.85
    tbl_fs = ax_title_fs * 0.75
    plt.rc('font', size=tbl_fs)

    if thickness > 0:
        fig_ohmic = plt.figure()
        plt.rc('font', size=14)
        ax1 = fig_ohmic.add_subplot(111)
        axy2 = ax1.twiny()
        ax1.plot(tk_1000,ah_cond,'o',markersize=10, color=txt_spine_color)

        # - Aligning the top axes tick marks with the bottom and converting to celcius
        def tick_function(X):
            V = (1000/np.array(X))-273
            return ["%.0f" % z for z in V]
        axy2.set_xticks(tk_1000)
        axy2.set_xticklabels(tick_function(tk_1000))

        # - Creating and Aligning the right axes with the left axes:
        axx2 = axy2.twinx()
        def tick_function(X):
            V = np.exp(np.array(X))/(273+np.array(temps))
            return ["%.4f" % z for z in V]
        axx2.set_yticks(ah_cond)
        axx2.set_yticklabels(tick_function(ah_cond))

        # - linear Fit:
        m, b, r, p_value, std_err = scipy.stats.linregress(tk_1000, ah_cond)
        plt.plot(tk_1000, m*tk_1000+b,'r',lw=spine_thickness*1.6)

        # - creating and formatting table:
        row_labels = ['Intercept','Slope','r squared']
        decimals = 2
        table_values = [[round(b,decimals)],[round(m,decimals)],[round(r**2,decimals+1)]]
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower center',rowColours= ['deepskyblue','deepskyblue','deepskyblue'])
        table.scale(1,1.6)

        # - Axis labels:
        ax1.set_xlabel('1000/T (1/K)')
        axy2.set_xlabel('Temperature (\u00B0C)')
        ax1.set_ylabel('ln(\u03C3T) (sK/cm)')
        axx2.set_ylabel('Rp ASR(\u03A9*$cm^2$)')

        # - Calculating and printing activation energy
        k = 8.617*10**-5 #boltzmanns constant in Ev/K
        Eact = round(m*k*(-1000),3) # this gives the activation energy in eV
        Eacts = f'{Eact}'
        fig_ohmic.text(0.33,0.33,r'$E_a$ ='+Eacts+'eV')

        plt.tight_layout()

    elif thickness == 0:
        fig_ohmic = plt.figure()
        
        ax1 = fig_ohmic.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(tk_1000,ah_ohmic_asr,'o',markersize=10, color=txt_spine_color)

        # - Aligning the top axes tick marks with the bottom and converting to celcius
        def tick_function(X):
            V = (1000/X)-273
            return ["%.0f" % z for z in V]
        ax2.set_xticks(tk_1000)
        ax2.set_xticklabels(tick_function(tk_1000),fontsize=ax_tl_fs)

        # # - Creating and Aligning the right axes with the left axes:
        # axy2 = ax2.twinx()
        # def tick_function(X):
        #     V = np.exp(np.array(X)) * (273+np.array(temps))
        #     return ["%.2f" % z for z in V]
        # axy2.set_yticks(ah_ohmic_asr)
        # axy2.set_yticklabels(tick_function(ah_ohmic_asr),fontsize=ax_tl_fs)
        # axy2.set_ylabel('R$_\mathrm{ohmic}$ (\u03A9 cm$^2$)', fontsize = ax_title_fs)

        # - linear Fit:
        m, b, r, p_value, std_err = scipy.stats.linregress(tk_1000, ah_ohmic_asr)
        plt.plot(tk_1000, m*tk_1000+b,'r',lw=spine_thickness*1.6)

        # - Creating and formatting table:
        row_labels = ['Intercept','Slope','r squared']
        decimals = 2
        table_values = [[round(b,decimals)],[round(m,decimals)],[round(r**2,decimals+1)]]
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',rowColours= ['lightblue','lightblue','lightblue'])
        table.scale(1,1.6)

        # - Axis labels:
        ax1.set_xlabel('1000/T (1/K)', fontsize = ax_title_fs)
        ax2.set_xlabel('Temperature (\u00B0C)', fontsize = ax_title_fs)
        ax1.set_ylabel('ln(ASR$_\mathrm{ohmic}$/T) (\u03A9 cm$^2$/K)', fontsize = ax_title_fs)

        # - Excessive formatting:
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', which='major', labelsize=ax_tl_fs)

        # - Calculating and printing activation energy
        k = 8.617*10**-5 #boltzmanns constant in Ev/K
        Eact = round(m*k*(1000),3) # this gives the activation energy in eV
        Eacts = f'{Eact}'
        fig_ohmic.text(0.69,0.35,r'$E_a$ ='+Eacts+'eV',fontsize=ax_tl_fs) 
        # Set to 0.6 if including ohmic axis 0.68 if not

        plt.tight_layout()

        # - saving the Ohmic Ahp plot
        if save_figures is not None:
            ohmic_fig_name = os.path.join(save_figures,'Ohmic_Ahp.png')
            fmat = ohmic_fig_name.split('.', 1)[-1]
            fig_ohmic.savefig(ohmic_fig_name, dpi=300, format=fmat, bbox_inches='tight') 

    ' --- Making the Polarizatation Resistance Arrhenius Plot --- '
    if rp_plt_type == 'asr':
        x = tk_1000
        y = rp_asr
        fig_rp = plt.figure()
        ax1 = fig_rp.add_subplot(111)
        ax2 = ax1.twiny() #creates a new x axis that is linked to the first y axis
        new_tick_locs = x 
        ax1.semilogy(x,y,'o',markersize=10, color=txt_spine_color)
        
        def tick_function(X):
            V = (1000/X)-273
            return ["%.0f" % z for z in V]
        ax2.set_xticks(new_tick_locs)
        ax2.set_xticklabels(tick_function(new_tick_locs))
        
        # - linear Fit:
        m, b, r, p_value, std_err = scipy.stats.linregress(x, np.log10(y))
        plt.plot(x, 10**(m*x+b),'r',lw=spine_thickness*1.6)
        
        # - creating table:
        row_labels = ['Intercept','Slope','r squared']
        decimals = 2
        table_values = [[round(b,decimals)],[round(m,decimals)],[round(r**2,decimals+1)]]
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',rowColours= ['gold','gold','gold'])
        table.scale(1,1.6)
        
        # - Axis labels:
        ax1.set_xlabel('1000/T (1/K)', fontsize = ax_title_fs)
        ax2.set_xlabel('Temperature (\u00B0C)', fontsize = ax_title_fs)
        ax1.set_ylabel('Rp ASR(\u03A9*$cm^2$)', fontsize = ax_title_fs)

        plt.tight_layout()

        # - saving the Rp Ahp plot
        if save_figures is not None:
            rp_fig_name = os.path.join(save_figures,'Rp_Ahp.png')
            fmat = rp_fig_name.split('.', 1)[-1]
            fig_rp.savefig(rp_fig_name, dpi=300, format=fmat, bbox_inches='tight') 

    elif rp_plt_type == 'ln':
        x = tk_1000
        y = ah_rp

        # - Initializing figure and plotting
        fig_rp = plt.figure()
        ax1 = fig_rp.add_subplot(111)
        ax2 = ax1.twiny() #creates a new x axis that is linked to the first y axis
        new_tick_locs = x #np.array([1.114,1.145,1.179,1.22,1.253,1.294])
        ax1.plot(x,y,'o',markersize=10, color=txt_spine_color)
        
        # - Adjusting ax2 ticks
        def tick_function(X):
            V = (1000/X)-273
            return ["%.0f" % z for z in V]
        ax2.set_xticks(new_tick_locs)
        ax2.set_xticklabels(tick_function(new_tick_locs),fontsize=ax_tl_fs)
        
        # - linear Fit:
        m, b, r, p_value, std_err = scipy.stats.linregress(x, y)
        plt.plot(x, (m*x+b),'r',lw=spine_thickness*1.6)
        
        # - creating table:
        row_labels = ['Intercept','Slope','r squared']
        decimals = 2
        table_values = [[round(b,decimals)],[round(m,decimals)],[round(r**2,decimals+1)]]
        table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',rowColours= ['gold','gold','gold'])
        table.scale(1,1.6)
        
        # - Axis labels:
        ax1.set_xlabel('1000/T (1/K)', fontsize=ax_title_fs)
        ax2.set_xlabel('Temperature (\u00B0C)', fontsize=ax_title_fs)
        ax1.set_ylabel('ln(ASR$_p$/T) (\u03A9 cm$^2$/K)', fontsize=ax_title_fs)

        # - Excessive formatting:
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', which='major', labelsize=ax_tl_fs)
        
        # - Calculating and printing activation energy
        k = 8.617*10**-5 #boltzmanns constant in Ev/K
        Eact = round(m*k*(1000),3) # this gives the activation energy in eV
        Eacts = f'{Eact}'
        fig_rp.text(0.68,0.35,r'$E_a$ ='+Eacts+'eV',fontsize=ax_tl_fs)

        plt.tight_layout()

        # - saving the Rp Ahp plot
        if save_figures is not None:
            rp_fig_name = os.path.join(save_figures,'Rp_Ahp.png')
            fmat = rp_fig_name.split('.', 1)[-1]
            fig_rp.savefig(rp_fig_name, dpi=300, format=fmat, bbox_inches='tight') 

    plt.show()

    ' --- Plotting EIS ---- '
    if plot_eis == True:
        # --- Setting up the color map
        cmap = plt.cm.get_cmap('coolwarm') #cmr.redshift 
        color_space = np.linspace(0,1,len(ahp_eis)) # array from 0-1 for the colormap for plotting
        c = 0 # indicie of the color array

        # --- Plotting
        for eis in reversed(ahp_eis):
            label = str(temps[len(ahp_eis)-c-1])+ ' \u00b0C'#r'$^\circ$C'
            color = cmap(color_space[c])
            plot_peiss(area,label,os.path.join(folder_loc,eis),color=color,legend_loc=legend_loc)
            c = c+1

        # - saving the EIS plots
        if save_figures is not None:
            eis_fig_name = os.path.join(save_figures,'Ahp_EIS.png')
            fmat = eis_fig_name.split('.', 1)[-1]
            plt.savefig(eis_fig_name, dpi=300, format=fmat, bbox_inches='tight') 

        plt.show()

    ' --- DRT peak fitting and plotting --- '
    if drt_peaks == True:
        # --- Checking to see if the peaks have already been fit:
        # if exists == True:
        peak_data_sheet = 'Ahp_' + drt_model + '_DRT_peaks'
        df_tau_r = pd.read_excel(excel_file,peak_data_sheet)

        # ----- plotting
        palette = sns.color_palette("coolwarm", as_cmap=True)
        plot = sns.scatterplot(x = 'Tau', y = 'Resistance', data = df_tau_r, hue='Temperature (C)',palette = palette,s=69)

        # -- Astetic stuff
        sns.set_context("talk")
        fontsize = 14
        sns.despine()
        
        plot.set_ylabel('ASR (\u03A9 cm$^2$)',fontsize=fontsize)
        plot.set_xlabel('Time Constant (\u03C4/s)',fontsize=fontsize)
        plot.set(xscale='log')
        
        if drtp_leg_loc_ots == True:
            plot.legend(loc='upper right',bbox_to_anchor=(1.05,1.0),fontsize=fontsize)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        plt.tight_layout()
        plt.show()

def arrhenius_iv_curves(folder_loc:str, area:float, temps:list, reverse:bool=False, 
                        leg_cols:int=None, save_plot:str=None):
    '''
    Searches through the folder_loc for hte IV curves taken during arrhenius testing. 
    It plots the iv curve and its corresponding power density curve for each temperature

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the directory containing the EIS files. 
    area, float:
        The active cell area of the cell in cm^2
    temps, list: 
        List of temperatures that the EIS spectra were taken at. this is input by the user 
        and must be in the same order that the EIS spectra were taken at.
    reverse, bool: optional (default: False)
        If the arrhenius plot was taken in reverse order i.e 500-625C (like my older cells), then set this to true
        This reverses the direction of the cmap to keep the lower temps blue and higher temps red on the DRT and Nyquist plots
        To reiterate if the arrhenisu plot was taken from lowest temp to highest set this to true
    leg_cols,int: (default is None)
        how many columns the legend will have. 
        If none, this will be based of the abount of curves being plot
        if leg_cols is set to 0, no legend is plot
        This is passed to the plot_ivfcs function
    save_plot, str: (default = None)
        If this is not none, the plot will be saved.
        Save_plot is the file name and path of the saved file.
        This is passed to the plot_ivfcs function

    Returns --> Nothing, but plots and shows the IV curves and power density curves for each temperature
    '''

    # --- Finding correct files and sorting
    ahp_iv = [file for file in os.listdir(folder_loc) if file.endswith('.DTA') and file.find('Ahp')!=-1 and file.find('IV')!=-1] #Makes a list of all ahp
    ahp_iv = sorted(ahp_iv, key=lambda x: int((x[x.find('Ahp_#')+len('Ahp_#'):x.rfind('.DTA')]))) #Sorts numerically by bias
    ahp_iv_loc = []
    for file in ahp_iv:
        ahp_iv_loc.append(os.path.join(folder_loc,file))

    area_list = [area]*len(ahp_iv)

    # --- Adding the degree symbol to the curves
    str_temps = []
    for temp in temps:
        temp = str(temp)
        temp += '\u00B0C'
        str_temps.append(temp)

    # --- Merging curves and conditions
    curves_conditions = tuple(zip(area_list,str_temps,ahp_iv_loc))

    if reverse == False:
        cmap = plt.cm.get_cmap('coolwarm_r')
    if reverse == True:
        cmap = plt.cm.get_cmap('coolwarm')

    plot_ivfcs(curves_conditions,print_Wmax=True,cmap=cmap,
               ppd_y_axis=True,leg_cols=leg_cols,font_size=24,
               save_plot = save_plot)

def arrhenius_dual_drt_peak(folder_loc:str, tau_low:float, tau_high:float, temps:np.array = None,
                        rmv_temp_r:np.array = None, rmv_temp_l:np.array = None, drt_model:str = 'dual',
                        print_c:bool = True):
    '''
    This function is meant to linearly fit a single DRT peak across a temperature range.
    After you use the arrhenius_plots function to plot the DRT peaks a cluster of peaks can be fit
    (this must be done for this function to work)

    Parameters
    ----------
    folder_loc, str: (path to a directory)
        The location of the directory containing the EIS files.
    tau_low, float:
        The lower bound of the time constant range to fit.
    tau_high, float:
        The upper bound of the time constant range to fit.
    temps, np.array: (default = None)
        The temperatures that the DRT peaks were taken at. if this is none all temperature ranges in
        the cell data sheet will be fit.
    rmv_temp_r, np.array: (default = None)
        If two clusters overlap, specify the temperatures where there are overlap and this will remove
        the peaks with higher time constants (lower frequency, to the right) from the fit.
    rmv_temp_l, np.array: (default = None)
        If two clusters overlap, specify the temperatures where there are overlap and this will remove
        the peaks with lower time constants (higher frequency, to the left) from the fit.
    drt_model, str: (default = dual)
        which type of drt used to analyze the data
        if drt = 'dual' the dual regression model is used to fit the data
        if drt = 'drtdop' then the drtdop model is used to fit the data
    print_c , bool: (default = False)
        if desired, the capacitance of each peak is calculated and fit

    Returns --> none, but a plot of the DRT peak fit and the activation energy is calculated and printed on the plot
    '''

    # ----- Sorting out the points from a specific time constant
    for file in os.listdir(folder_loc):
        if file.endswith('Data.xlsx'):
            data_file = os.path.join(folder_loc,file)
            break

    sheet_name = 'Ahp_' + drt_model + '_DRT_peaks'

    data = pd.read_excel(data_file, sheet_name)
    data = data[(data['Tau']>tau_low) & (data['Tau']<tau_high)]

    # ----- If desired, only plotting certain temperatures
    if temps is not None:
        data = data[data['Temperature (C)'].isin(temps)]

    # ----- removing a duplicate if there is overlap between clusters to the left (remove points to the left)
    if rmv_temp_l is not None:
        for temp in rmv_temp_l:
            # - find the lowest tau for a given temperature
            temp_data = data[data['Temperature (C)']==temp]
            temp_data = temp_data.sort_values(by='Tau')

            try:
                lowest_tau = temp_data.iloc[0]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed temperature must be in temps/the range of temperatures plotted.')
                print('Check the temps array and the rmv_dup_temps array')
                sys.exit(1)

            # - remove all higher tau values
            # data = data[data['Tau']!=lowest_tau]
            data = data[~((data['Temperature (C)'] == temp) & (data['Tau'] == lowest_tau))]

    # ----- removing a duplicate if there is overlap between clusters to the right (remove points to the right)
    if rmv_temp_r is not None:
        for temp in rmv_temp_r:
            # - find the lowest tau for a given temperature
            temp_data = data[data['Temperature (C)']==temp]
            temp_data = temp_data.sort_values(by='Tau')

            try:
                highest_tau = temp_data.iloc[-1]['Tau']

            except IndexError as error:
                traceback.print_exc()
                print('The removed temperature must be in temps/the range of temperatures plotted.')
                print('Check the temps array and the rmv_temp array')
                sys.exit(1)

            # - remove all higher tau values
            # data = data[data['Tau']!=highest_tau]
            data = data[~((data['Temperature (C)'] == temp) & (data['Tau'] == highest_tau))]

    # ----- Finding low and high Tau values of the values plotted
    min_tau = data['Tau'].min()
    max_tau = data['Tau'].max()

    # ----- Formatting data for plotting
    rp_asr = data['Resistance'].values
    temps = data['Temperature (C)'].values
    ah_rp = np.log(rp_asr/(273+np.array(temps)))
    
    ah_temps = np.array(temps)
    ah_temps = 1000/(ah_temps + 273)

    # --- If printing capacitance is desired
    data['Capacitance (F/cm^2)'] = data['Tau']/data['Resistance']
    if print_c == True:
        print(data)
        
    # ----- Plotting
    x = ah_temps
    y = ah_rp
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax1.plot(x,y,'ko')

    # - Setting fontsizes
    label_fontsize = 'x-large'
    tick_fontsize = 'large'
    text_fontsize = 'x-large'

    # - Setting ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(temps,fontsize=tick_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(np.round(x,3),fontsize=tick_fontsize)
    ax1.set_yticks(y)
    ax1.set_yticklabels(np.round(y,2),fontsize=tick_fontsize)

    # - linear Fit:
    m, b, r, p_value, std_err = scipy.stats.linregress(x, y)
    plt.plot(x, (m*x+b),'r')

    # - Creating table:
    row_labels = ['Intercept','Slope','r squared']
    table_values = [[round(b,3)],[round(m,3)],[round(r**2,3)]]
    table = plt.table(cellText=table_values,colWidths = [.2]*3,rowLabels=row_labels,loc = 'lower right',
        rowColours= ['plum','plum','plum'])
    table.scale(1,1.6)

    # - Axis labels:
    ax1.set_xlabel('1000/T (1/K)',fontsize=label_fontsize)
    ax2.set_xlabel('Temperature (\u00B0C)',fontsize=label_fontsize)
    ax1.set_ylabel('ln(ASR/T) (\u03A9 cm$^2$/K)',fontsize=label_fontsize)
    
    # - Calculating activation energy
    k = 8.617*10**-5 #boltzmanns constant in Ev/K
    Eact = round(m*k*(1000),3) # this gives the activation energy in eV
    
    # ----- Printing figure text
    # - Activation Energy
    Eacts = f'{Eact}'
    fig.text(0.72,0.33,r'$E_a$ ='+Eacts+'eV',fontsize = text_fontsize)
    # - Time Constants
    tau_lows = f'{min_tau:.2e}'
    tau_highs = f'{max_tau:.2e}'
    fig.text(0.18,0.81,'DRT peak between '+tau_lows+'(\u03C4/s) and '+tau_highs+'(\u03C4/s)',fontsize = tick_fontsize)
    # - Capacitances
    avg_c = data['Capacitance (F/cm^2)'].mean()
    std_c = data['Capacitance (F/cm^2)'].std()
    avg_cs = f'{avg_c:.2e}'
    std_cs = f'{std_c:.2e}'
    fig.text(0.18,0.76,'Avg C: '+avg_cs+r'(F/cm$^2$) +/- '+std_cs,fontsize = tick_fontsize)


    plt.tight_layout()

    plt.show()



